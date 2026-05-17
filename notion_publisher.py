#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NotionPublisher - 将 VideoSummary 生成的 Markdown 笔记推送到 Notion。

用法:
    from notion_publisher import NotionPublisher

    publisher = NotionPublisher(token="ntn_xxx", parent_page_id="xxx")
    publisher.push_markdown("output/xxx.md", title="CMU 15-445",
                            github_raw_base="https://raw.githubusercontent.com/.../main")
"""

import os
import re
import logging
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote as url_quote

logger = logging.getLogger(__name__)

# Notion blocks 每批最多 100 个
MAX_BLOCKS_PER_REQUEST = 100


# ---- Markdown 文本装饰解析 ----
def _parse_inline_rich_text(text: str) -> List[dict]:
    """
    解析一行 Markdown 文本中的内联样式（粗体、斜体、行内代码），
    返回 Notion rich_text 数组。

    例如: "这是 **粗体** 和 `代码`"
    → [text("这是 "), text("粗体", bold=True), text(" 和 "), text("代码", code=True)]
    """
    if not text:
        return [{"type": "text", "text": {"content": ""}}]

    tokens: List[dict] = []
    # 匹配顺序很重要：代码 > 粗体 > 斜体
    pattern = re.compile(
        r'(?P<code>`[^`]+`)'
        r'|(?P<bold>\*\*[^*]+\*\*)'
        r'|(?P<italic>\*[^*]+\*)'
        r'|(?P<plain>[^`*]+)'
    )

    for m in pattern.finditer(text):
        if m.lastgroup == "code" and m.group("code"):
            code_content = m.group("code")[1:-1]  # 去掉反引号
            tokens.append({
                "type": "text",
                "text": {"content": code_content},
                "annotations": {"code": True},
            })
        elif m.lastgroup == "bold" and m.group("bold"):
            bold_content = m.group("bold")[2:-2]  # 去掉 ** **
            tokens.append({
                "type": "text",
                "text": {"content": bold_content},
                "annotations": {"bold": True},
            })
        elif m.lastgroup == "italic" and m.group("italic"):
            italic_content = m.group("italic")[1:-1]  # 去掉 * *
            tokens.append({
                "type": "text",
                "text": {"content": italic_content},
                "annotations": {"italic": True},
            })
        elif m.lastgroup == "plain" and m.group("plain"):
            tokens.append({
                "type": "text",
                "text": {"content": m.group("plain")},
            })

    if not tokens:
        tokens = [{"type": "text", "text": {"content": text}}]

    return tokens


def _make_rich_text(text: str) -> List[dict]:
    """将纯文本转为 Notion rich_text，不做样式解析。"""
    if not text:
        return [{"type": "text", "text": {"content": ""}}]
    return [{"type": "text", "text": {"content": text}}]


def _make_rich_text_inline(text: str) -> List[dict]:
    """解析 Markdown 内联样式，返回带样式的 rich_text。"""
    return _parse_inline_rich_text(text) if text else _make_rich_text("")


# ---- Markdown Block 转换 ----
def _is_ordered_list_item(line: str) -> bool:
    """判断是否是有序列表项（以 数字.  开头）"""
    stripped = line.strip()
    return bool(re.match(r'^\d+\.\s', stripped))


def _is_unordered_list_item(line: str) -> bool:
    """判断是否是无序列表项（以 - 或 * 开头，后跟空格）"""
    stripped = line.strip()
    return len(stripped) >= 2 and stripped[0] in ('-', '*') and stripped[1] == ' '


def _is_any_list_item(line: str) -> bool:
    """判断是否是任意类型的列表项"""
    return _is_ordered_list_item(line) or _is_unordered_list_item(line)


def _get_indent_level(line: str) -> int:
    """返回行的缩进级别（每 2 个空格算一级）"""
    spaces = len(line) - len(line.lstrip(' '))


def _get_indent_level(line: str) -> int:
    """返回行的缩进级别（每 2 个空格算一级）"""
    spaces = len(line) - len(line.lstrip(' '))
    return spaces // 2


def _strip_list_prefix(line: str) -> str:
    """去掉列表前缀（`1. `, `- `, `* ` 等），并清理后续多余空格"""
    stripped = line.strip()
    # 有序列表: 1. text
    m = re.match(r'^\d+\.\s+(.*)', stripped)
    if m:
        return m.group(1)
    # 无序列表: - text 或 * text
    if len(stripped) >= 2 and stripped[0] in ('-', '*') and stripped[1] == ' ':
        return stripped[2:].lstrip()
    return stripped


def _is_table_row(line: str) -> bool:
    """判断是否是 Markdown 表格行（包含 | 分隔符）"""
    s = line.strip()
    return s.startswith('|') and '|' in s[1:]


def _is_table_separator(line: str) -> bool:
    """判断是否是表格分隔行 `|:---|:---|`"""
    s = line.strip()
    if not (s.startswith('|') and s.endswith('|')):
        return False
    # 分隔行只包含 |, -, :, 空格
    inner = s[1:-1]
    return bool(re.match(r'^[\s\-:|]+$', inner))


def _make_image_block(image_url: str) -> dict:
    """创建 Notion image block（external URL）。"""
    return {
        "object": "block",
        "type": "image",
        "image": {
            "type": "external",
            "external": {"url": image_url},
        },
    }


def _is_divider(line: str) -> bool:
    """判断是否是分隔线 `---` 或 `***`"""
    stripped = line.strip()
    return re.match(r'^[-*]{3,}$', stripped) is not None


def _is_code_fence(line: str) -> bool:
    """判断是否以 ``` 开头或结尾"""
    return line.strip().startswith('```')


# ---- NotionPublisher ----
class NotionPublisher:
    """
    将 Markdown 笔记推送到 Notion。
    """

    def __init__(self, token: str, parent_page_id: str):
        """
        Args:
            token: Notion Integration Token（格式 ntn_xxx 或 secret_xxx）
            parent_page_id: 父页面 ID（笔记会作为子页面创建）
        """
        self.token = token.strip()
        self.parent_page_id = parent_page_id.strip().replace('-', '')

        if not self.token or not self.parent_page_id:
            raise ValueError("NOTION_TOKEN 和 NOTION_PARENT_PAGE_ID 不能为空")

        # 延迟导入，避免未安装时报错
        try:
            from notion_client import Client
        except ImportError:
            raise ImportError(
                "未安装 notion-client，请执行: pip install notion-client")

        self.client = Client(auth=self.token)
        logger.info("🔗 Notion client 已初始化")

    def push_markdown(
        self,
        md_path: str,
        title: str,
        github_user: str,
        github_repo: str,
        github_branch: str = "main",
    ) -> Optional[str]:
        """
        读取 Markdown 文件，转换为 Notion blocks 并创建页面。

        Args:
            md_path: Markdown 文件路径
            title: 笔记标题（也作为 Notion 页面标题）
            github_user: GitHub 用户名
            github_repo: GitHub 仓库名
            github_branch: 分支名

        Returns:
            创建的 Notion 页面 URL，失败返回 None
        """
        if not os.path.exists(md_path):
            logger.error(f"❌ 文件不存在: {md_path}")
            return None

        with open(md_path, 'r', encoding='utf-8') as f:
            md_text = f.read()

        logger.info(f"📝 解析 Markdown: {os.path.basename(md_path)} ({len(md_text)} 字符)")

        # 构建 GitHub raw URL 前缀
        raw_base = f"https://raw.githubusercontent.com/{github_user}/{github_repo}/{github_branch}"

        # 转换 Markdown → Notion blocks
        blocks = self._md_to_blocks(md_text, raw_base)
        logger.info(f"  转换完成: {len(blocks)} 个 Notion blocks")

        if not blocks:
            logger.warning("⚠️  没有可转换的内容")
            return None

        # 创建页面
        try:
            page_url = self._create_page_with_blocks(title, blocks)
            logger.info(f"✅ Notion 页面已创建: {page_url}")
            return page_url
        except Exception as e:
            logger.error(f"❌ 推送 Notion 失败: {e}")
            return None

    def _md_to_blocks(self, md_text: str, raw_base: str) -> List[dict]:
        """
        将 Markdown 文本转换为 Notion block 数组。
        支持：标题(h1-h4)、段落、粗体/斜体/行内代码、无序列表（嵌套）、
              代码块、表格（渲染为代码块）、分隔线、图片、引用。

        Args:
            md_text: Markdown 原文
            raw_base: GitHub raw URL 前缀
        """
        lines = md_text.split('\n')
        blocks: List[dict] = []

        # ---- 嵌套列表处理器 ----
        # 每个元素: {"indent": int, "node": dict}
        # node 是 Notion bulleted_list_item block dict（含 children）
        flat_list_nodes: List[dict] = []  # [(indent, block_dict)]

        def _make_list_block(text: str, ordered: bool = False) -> dict:
            block_type = "numbered_list_item" if ordered else "bulleted_list_item"
            return {
                "object": "block",
                "type": block_type,
                block_type: {
                    "rich_text": _make_rich_text_inline(text),
                },
            }

        # Notion API 限制：bulleted_list_item 最多嵌套 2 层
        MAX_LIST_NEST_DEPTH = 2

        def _nest_list_nodes(nodes: List[dict]) -> List[dict]:
            """
            将扁平列表节点按缩进层级嵌套为 Notion children 结构。
            Notion API 限制最多 2 层嵌套（root → child → grandchild），
            超出部分会被提升为上级兄弟节点。

            nodes: [{"indent": int, "node": dict}, ...]
            返回: 顶层 block 列表（含嵌套 children）
            """
            if not nodes:
                return []

            result: List[dict] = []
            stack: List[Tuple[int, dict]] = []

            for item in nodes:
                indent = item["indent"]
                node = item["node"]

                # 限制缩进深度：超过 MAX_LIST_NEST_DEPTH 的提升
                capped_indent = min(indent, MAX_LIST_NEST_DEPTH)

                if not stack:
                    result.append(node)
                    stack.append((capped_indent, node))
                    continue

                # 找到合适的父级
                while stack and stack[-1][0] >= capped_indent:
                    stack.pop()

                if stack:
                    parent = stack[-1][1]
                    # 获取父级 block 的内部 dict（支持两种列表类型）
                    parent_type = parent.get("type", "")
                    parent_block = parent.get(parent_type, parent)
                    if "children" not in parent_block:
                        parent_block["children"] = []
                    parent_block["children"].append(node)
                else:
                    result.append(node)

                stack.append((capped_indent, node))

            return result

        # ---- 状态变量 ----
        i = 0
        pending_paragraph: List[str] = []
        pending_table_rows: List[str] = []

        def flush_list():
            """将收集的扁平列表节点嵌套后写入 blocks"""
            nonlocal flat_list_nodes
            if flat_list_nodes:
                nested = _nest_list_nodes(flat_list_nodes)
                blocks.extend(nested)
                flat_list_nodes = []

        def flush_paragraph():
            nonlocal pending_paragraph
            if pending_paragraph:
                combined = '\n'.join(pending_paragraph).strip()
                if combined:
                    blocks.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": _make_rich_text_inline(combined),
                        },
                    })
                pending_paragraph = []

        def flush_table():
            """将收集的表格行渲染为 code block"""
            nonlocal pending_table_rows
            if pending_table_rows:
                table_text = '\n'.join(pending_table_rows)
                blocks.append({
                    "object": "block",
                    "type": "code",
                    "code": {
                        "language": "plain text",
                        "rich_text": [{"type": "text", "text": {"content": table_text}}],
                    },
                })
                pending_table_rows = []

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # ---- 空行 ----
            if not stripped:
                flush_list()
                flush_paragraph()
                flush_table()
                i += 1
                continue

            # ---- 表格行 ----
            if _is_table_row(stripped):
                flush_list()
                flush_paragraph()
                pending_table_rows.append(stripped)
                i += 1
                continue
            elif _is_table_separator(stripped):
                # 跳过分隔行，但继续收集中（不中断表格）
                i += 1
                continue
            elif pending_table_rows:
                # 上一行是表格，当前不是 → 表格结束
                flush_table()

            # ---- 一级标题 ----
            if stripped.startswith('# ') and not stripped.startswith('## '):
                flush_list()
                flush_paragraph()
                blocks.append({
                    "object": "block", "type": "heading_1",
                    "heading_1": {"rich_text": _make_rich_text(stripped[2:])},
                })
                i += 1
                continue

            # ---- 二级标题 ----
            if stripped.startswith('## ') and not stripped.startswith('### '):
                flush_list()
                flush_paragraph()
                blocks.append({
                    "object": "block", "type": "heading_2",
                    "heading_2": {"rich_text": _make_rich_text(stripped[3:])},
                })
                i += 1
                continue

            # ---- 三级标题 ----
            if stripped.startswith('### ') and not stripped.startswith('#### '):
                flush_list()
                flush_paragraph()
                blocks.append({
                    "object": "block", "type": "heading_3",
                    "heading_3": {"rich_text": _make_rich_text(stripped[4:])},
                })
                i += 1
                continue

            # ---- 四级标题（Notion 无 h4，映射为 h3） ----
            if stripped.startswith('#### '):
                flush_list()
                flush_paragraph()
                blocks.append({
                    "object": "block", "type": "heading_3",
                    "heading_3": {"rich_text": _make_rich_text(stripped[5:])},
                })
                i += 1
                continue

            # ---- 代码块 ----
            if _is_code_fence(stripped):
                flush_list()
                flush_paragraph()
                i += 1
                code_lines: List[str] = []
                lang = stripped[3:].strip() or "plain text"
                while i < len(lines) and not _is_code_fence(lines[i]):
                    code_lines.append(lines[i])
                    i += 1
                i += 1
                blocks.append({
                    "object": "block", "type": "code",
                    "code": {
                        "language": lang,
                        "rich_text": _make_rich_text('\n'.join(code_lines)),
                    },
                })
                continue

            # ---- 分隔线 ----
            if _is_divider(stripped):
                flush_list()
                flush_paragraph()
                blocks.append({"object": "block", "type": "divider", "divider": {}})
                i += 1
                continue

            # ---- 图片 ----
            if stripped.startswith('!['):
                flush_list()
                flush_paragraph()
                img_match = re.match(r'!\[.*?\]\((.+?)\)', stripped)
                if img_match:
                    blocks.append(_make_image_block(
                        self._resolve_image_url(img_match.group(1), raw_base)))
                i += 1
                continue

            # ---- 列表项（支持有序/无序、嵌套缩进） ----
            if _is_any_list_item(stripped):
                flush_paragraph()
                item_text = _strip_list_prefix(stripped)
                is_ordered = _is_ordered_list_item(stripped)

                line_indent = _get_indent_level(line)

                flat_list_nodes.append({
                    "indent": line_indent,
                    "node": _make_list_block(item_text, ordered=is_ordered),
                })
                i += 1
                continue

            # ---- 引用块（> 开头） ----
            if stripped.startswith('> '):
                flush_list()
                flush_paragraph()
                blocks.append({
                    "object": "block", "type": "quote",
                    "quote": {"rich_text": _make_rich_text_inline(stripped[2:])},
                })
                i += 1
                continue

            # ---- 普通文本（段落） ----
            flush_list()
            pending_paragraph.append(stripped)
            i += 1

        # 收尾
        flush_list()
        flush_paragraph()
        flush_table()

        # 清理空段落
        blocks = [b for b in blocks if b.get("type") != "paragraph"
                   or any(t.get("text", {}).get("content", "").strip()
                          for t in b.get("paragraph", {}).get("rich_text", [])
                          if t.get("type") == "text")]

        return blocks

    def _resolve_image_url(self, local_path: str, raw_base: str) -> str:
        """
        将本地相对路径转为 GitHub raw URL。

        Args:
            local_path: Markdown 中的相对路径，如 "xxx_frames/chunk_01/img.jpg"
            raw_base: GitHub raw URL 前缀

        Returns:
            完整的公网 URL
        """
        # URL encode 路径中的特殊字符（中文、空格等）
        encoded_path = url_quote(local_path, safe='/')
        # 去掉可能的 ./ 前缀
        if encoded_path.startswith('./'):
            encoded_path = encoded_path[2:]

        return f"{raw_base}/{encoded_path}"

    def _create_page_with_blocks(
        self, title: str, blocks: List[dict]
    ) -> str:
        """
        在父页面下创建子页面，并写入所有 blocks。
        超过 100 个 block 时分批追加。

        Returns:
            创建的页面 URL
        """
        # 第一批 blocks（最多 100 个）随页面创建一起提交
        first_batch = blocks[:MAX_BLOCKS_PER_REQUEST]
        remaining = blocks[MAX_BLOCKS_PER_REQUEST:]

        logger.info(f"  📄 创建 Notion 页面: {title}")

        response = self.client.pages.create(
            parent={"page_id": self.parent_page_id},
            properties={
                "title": [{"type": "text", "text": {"content": title}}],
            },
            children=first_batch,
        )

        page_id = response["id"]
        page_url = response.get("url", f"https://notion.so/{page_id}")

        # 剩余的 blocks 分批追加
        offset = 0
        while offset < len(remaining):
            batch = remaining[offset:offset + MAX_BLOCKS_PER_REQUEST]
            logger.info(f"  📎 追加 blocks {offset + len(first_batch) + 1}...")
            self.client.blocks.children.append(
                block_id=page_id,
                children=batch,
            )
            offset += MAX_BLOCKS_PER_REQUEST

        if len(blocks) > MAX_BLOCKS_PER_REQUEST:
            logger.info(f"  ✅ 共写入 {len(blocks)} 个 blocks（分 {1 + (len(remaining) + MAX_BLOCKS_PER_REQUEST - 1) // MAX_BLOCKS_PER_REQUEST} 批）")

        return page_url
