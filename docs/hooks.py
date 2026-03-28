"""MkDocs hooks to customize documentation build.

Debug Logging
-------------
To enable debug logging for bullet list conversions, set the environment variable:
    export YUZUHA_HOOKS_DEBUG=1
    mkdocs build

Logs are written to: .logging/hooks.log
"""

import re
import os
from datetime import datetime


# Global counter for tracking hook invocations
_bullet_fix_counter = 0
_log_file_path = os.path.join(os.path.dirname(__file__), '.logging', 'hooks.log')

# Logging switch - set environment variable YUZUHA_HOOKS_DEBUG=1 to enable
_debug_logging = os.environ.get('YUZUHA_HOOKS_DEBUG', '0') == '1'


def _log(message):
    """Write a message to the log file (only if debug logging is enabled)."""
    if not _debug_logging:
        return
    os.makedirs(os.path.dirname(_log_file_path), exist_ok=True)
    with open(_log_file_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def on_post_page(output, page, config):
    """Post-process HTML to convert markdown bullet lists in tables to proper HTML lists.

    This runs after the page HTML is generated and converts any markdown-style
    bullet lists (- item) that appear in parameter tables to proper <ul>/<li> tags.
    """
    global _bullet_fix_counter

    # Initialize log for this build session
    if _debug_logging and _bullet_fix_counter == 0:
        print(f"[YUZUHA_HOOKS_DEBUG] Logging enabled - writing to {_log_file_path}")
        _log(f"\n{'='*80}")
        _log(f"Hook session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        _log(f"{'='*80}\n")

    def fix_table_bullets(match):
        """Convert markdown bullets to HTML list in table cell."""
        global _bullet_fix_counter

        content = match.group(1)

        if _debug_logging:
            _bullet_fix_counter += 1
            _log(f"\n{'-'*80}")
            _log(f"Match #{_bullet_fix_counter} - Page: {page.file.src_path}")
            _log(f"{'-'*80}")
            _log(f"Original content (first 300 chars):")
            _log(content[:300].replace('\n', '\\n'))
            if len(content) > 300:
                _log(f"... (total length: {len(content)} chars)")

        if '\n- ' not in content and not content.strip().startswith('-'):
            if _debug_logging:
                _log("No bullets detected - skipping")
            return match.group(0)

        lines = content.split('\n')
        new_lines = []
        in_list = False
        list_items = []
        preamble_lines = []

        for line in lines:
            line_text = re.sub(r'<[^>]+>', '', line)
            bullet_match = re.match(r'^\s*- (.+)$', line_text)

            is_continuation = (in_list and
                               line_text.strip() and
                               not bullet_match and
                               re.match(r'^\s+', line_text))

            if bullet_match:
                if not in_list and preamble_lines:
                    new_lines.extend(preamble_lines)
                    preamble_lines = []
                    in_list = True

                bullet_content = re.sub(r'^\s*- ', '', line)
                list_items.append(bullet_content.strip())
                in_list = True
            elif is_continuation:
                if list_items:
                    continuation_text = line.strip()
                    list_items[-1] += ' ' + continuation_text
            else:
                if in_list:
                    new_lines.append('<ul>')
                    for item in list_items:
                        new_lines.append(f'<li>{item}</li>')
                    new_lines.append('</ul>')
                    list_items = []
                    in_list = False

                if line.strip():
                    preamble_lines.append(line)

        if in_list and list_items:
            if preamble_lines:
                new_lines.extend(preamble_lines)
                preamble_lines = []
            new_lines.append('<ul>')
            for item in list_items:
                new_lines.append(f'<li>{item}</li>')
            new_lines.append('</ul>')
        elif preamble_lines:
            new_lines.extend(preamble_lines)

        result = '\n'.join(new_lines)

        if _debug_logging:
            _log(f"\nConverted to HTML (first 300 chars):")
            _log(result[:300].replace('\n', '\\n'))
            if len(result) > 300:
                _log(f"... (total length: {len(result)} chars)")

        return f'<p>{result}</p>'

    inline_content = r'(?:[^<\n]|</?(?:code|em|strong|a|span|b|i|u|sup|sub)(?:\s[^>]*)?>|\n)*?'
    output = re.sub(
        rf'<p>({inline_content}(?:\n- [^\n]+)+{inline_content})</p>',
        fix_table_bullets,
        output,
        flags=re.MULTILINE
    )

    return output


def on_post_build(config):
    """Hook called after the build is complete. Log final statistics."""
    global _bullet_fix_counter

    if _debug_logging:
        _log(f"\n{'='*80}")
        _log(f"Hook session completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        _log(f"Total bullet list conversions: {_bullet_fix_counter}")
        _log(f"{'='*80}\n")
        print(f"[YUZUHA_HOOKS_DEBUG] Session completed. Total conversions: {_bullet_fix_counter}")

    _bullet_fix_counter = 0
