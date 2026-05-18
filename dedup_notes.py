#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch dedup script for lecture notes.
Each lecture file was generated from multiple overlapping transcript chunks,
resulting in duplicate content across "## 第 N 部分" sections.
This script sends each file to DeepSeek for consolidation.

Usage:
    python dedup_notes.py output/CMU_15-445        # process one course
    python dedup_notes.py output/CS149              # process one course
    python dedup_notes.py --all                     # process all courses
"""

import os
import sys
import time
import argparse
import logging
from glob import glob
from typing import List, Tuple

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not found. Install with: pip install openai")
    sys.exit(1)

# --- Configuration ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
PRIMARY_MODEL = "deepseek-v4-pro"
MAX_RETRIES = 5
RETRY_DELAY = 30  # seconds

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Dedup Prompt ---
DEDUP_SYSTEM_PROMPT = """你是一名专业的技术编辑。一段课程视频被切分成多个带重叠的字幕片段，每个片段独立交给 AI 生成了笔记。因此，同一个知识点在多个"## 第 N 部分"中被反复解释（措辞不同但内容重复）。

你的任务：将这些零散的笔记**合并去重**，输出一份干净、无冗余的完整笔记。

## 规则

1. **合并重复章节**：当同一个概念在多个"部分"中出现时，合并为一个章节。保留最完整/最清晰的解释，删除重复部分。
2. **重组为逻辑章节**：用 `## 一、...`、`## 二、...` 等逻辑章节替代原始的 `## 第 N 部分` 标记。
3. **添加主题摘要**：在标题后添加 `> 主题：...` 行，概括本讲的中心内容。
4. **保留原标题**：`# 标题` 保持不变。
5. **保留格式**：保持所有项目符号、加粗术语、表格、LaTeX 公式、代码块原样。仅删除逐字重复的内容。
6. **精简冗长表述**：同一观点在附近多段重复时，保留最精炼的版本。
7. **保留所有图片引用**：所有 `![截图](...)` 引用必须完整保留，不要修改、删除或移动位置。
8. **只输出中文笔记**：最终输出为中文（与输入保持一致）。
9. **禁止元对话**：不要输出"以下是合并后的笔记"之类的说明。直接从内容开始。
10. **不添加新内容**：只重组和去重已有内容，不要发明新的解释或例子。

直接输出合并后的干净笔记。"""

# --- Courses configuration ---
COURSES = {
    "CMU_15-445": "output/CMU_15-445",
    "CS149": "output/CS149",
    "CS336": "output/CS336",
    "MIT_6.824": "output/MIT_6.824",
}


def count_parts(content: str) -> int:
    """Count the number of '## 第 N 部分' sections in the file."""
    import re
    return len(re.findall(r'## 第 \d+ 部分', content))


def dedup_file(client: OpenAI, filepath: str) -> Tuple[bool, int, int]:
    """
    Deduplicate a single file via DeepSeek.
    Returns (success, original_lines, new_lines).
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        logger.error(f"  Cannot read {filepath}: {e}")
        return (False, 0, 0)

    original_lines = len(original_content.splitlines())
    part_count = count_parts(original_content)

    if part_count <= 1:
        logger.info(f"  Skipping (only {part_count} part(s)): {os.path.basename(filepath)}")
        return (True, original_lines, original_lines)

    logger.info(f"  Processing: {os.path.basename(filepath)} "
                f"({original_lines} lines, {part_count} parts)")

    prompt = DEDUP_SYSTEM_PROMPT + "\n\n以下是需要合并去重的原始笔记：\n\n" + original_content

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # low temperature for consistent dedup
            )
            result = response.choices[0].message.content

            if not result or not result.strip():
                raise RuntimeError("LLM returned empty result")

            result = result.strip()
            new_lines = len(result.splitlines())

            # Write back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(result)
                if not result.endswith('\n'):
                    f.write('\n')

            reduction = (1 - new_lines / original_lines) * 100 if original_lines > 0 else 0
            logger.info(f"  Done: {os.path.basename(filepath)} "
                        f"{original_lines} → {new_lines} lines "
                        f"(-{reduction:.0f}%)")
            return (True, original_lines, new_lines)

        except Exception as e:
            logger.warning(f"  Attempt {attempt}/{MAX_RETRIES} failed for "
                           f"{os.path.basename(filepath)}: {e}")
            if attempt < MAX_RETRIES:
                logger.info(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"  All retries exhausted for {filepath}")
                return (False, original_lines, 0)


def process_course(client: OpenAI, course_dir: str) -> Tuple[int, int, int, int]:
    """
    Process all .md files in a course directory.
    Returns (total_files, success_files, total_original_lines, total_new_lines).
    """
    pattern = os.path.join(course_dir, "*_p*.md")
    files = sorted(glob(pattern))

    if not files:
        logger.warning(f"No *_p*.md files found in {course_dir}")
        return (0, 0, 0, 0)

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing course: {os.path.basename(course_dir)} ({len(files)} files)")
    logger.info(f"{'='*60}")

    total = len(files)
    success = 0
    total_orig = 0
    total_new = 0

    for i, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)
        logger.info(f"[{i}/{total}] {filename}")

        ok, orig, new = dedup_file(client, filepath)
        if ok:
            success += 1
            total_orig += orig
            total_new += new
        else:
            logger.error(f"  FAILED: {filename}")

        # Small delay between files to avoid rate limiting
        if i < total:
            time.sleep(1)

    if total > 0:
        reduction = (1 - total_new / total_orig) * 100 if total_orig > 0 else 0
        logger.info(f"\n{'='*60}")
        logger.info(f"Course summary: {os.path.basename(course_dir)}")
        logger.info(f"  Files: {success}/{total} processed successfully")
        logger.info(f"  Lines: {total_orig:,} → {total_new:,} (-{reduction:.0f}%)")
        logger.info(f"{'='*60}\n")

    return (total, success, total_orig, total_new)


def main():
    parser = argparse.ArgumentParser(
        description="Batch dedup lecture notes using DeepSeek API")
    parser.add_argument("course_dir", nargs="?", default=None,
                        help="Path to course directory (e.g., output/CMU_15-445)")
    parser.add_argument("--all", action="store_true",
                        help="Process all courses")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without processing")
    parser.add_argument("--file", type=str, default=None,
                        help="Process a single file")

    args = parser.parse_args()

    # Validate API key
    api_key = os.environ.get("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
    if not api_key or "YOUR_API_KEY" in api_key:
        logger.error("DEEPSEEK_API_KEY not set. "
                     "Please set the environment variable.")
        sys.exit(1)

    # Initialize client
    try:
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    except Exception as e:
        logger.error(f"Failed to initialize API client: {e}")
        sys.exit(1)

    logger.info(f"API configured: {DEEPSEEK_BASE_URL}, model: {PRIMARY_MODEL}")

    # Single file mode
    if args.file:
        filepath = args.file
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            sys.exit(1)
        if args.dry_run:
            content = open(filepath, 'r', encoding='utf-8').read()
            parts = count_parts(content)
            lines = len(content.splitlines())
            logger.info(f"Would process: {filepath} ({lines} lines, {parts} parts)")
            return
        ok, orig, new = dedup_file(client, filepath)
        if not ok:
            sys.exit(1)
        return

    # Determine which courses to process
    if args.all:
        course_dirs = list(COURSES.values())
    elif args.course_dir:
        course_dirs = [args.course_dir]
    else:
        parser.print_help()
        print("\nAvailable courses:")
        for name, path in COURSES.items():
            print(f"  {name}: {path}")
        sys.exit(1)

    # Dry run mode
    if args.dry_run:
        for course_dir in course_dirs:
            pattern = os.path.join(course_dir, "*_p*.md")
            files = sorted(glob(pattern))
            logger.info(f"\n{course_dir}: {len(files)} files")
            for f in files:
                content = open(f, 'r', encoding='utf-8').read()
                parts = count_parts(content)
                lines = len(content.splitlines())
                logger.info(f"  {os.path.basename(f)}: {lines} lines, {parts} parts")
        return

    # Process courses
    grand_total_files = 0
    grand_success = 0
    grand_orig = 0
    grand_new = 0

    for course_dir in course_dirs:
        total, success, orig, new = process_course(client, course_dir)
        grand_total_files += total
        grand_success += success
        grand_orig += orig
        grand_new += new

    # Grand summary
    if grand_total_files > 0:
        reduction = (1 - grand_new / grand_orig) * 100 if grand_orig > 0 else 0
        logger.info(f"\n{'='*60}")
        logger.info(f"GRAND TOTAL")
        logger.info(f"  Files: {grand_success}/{grand_total_files} processed")
        logger.info(f"  Lines: {grand_orig:,} → {grand_new:,} (-{reduction:.0f}%)")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
