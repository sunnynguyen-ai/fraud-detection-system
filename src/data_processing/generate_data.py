#!/usr/bin/env python3
"""
Complete script to fix all linting issues reported by flake8
Run this from the project root directory: /workspaces/fraud-detection-system/
"""

import os
import re
from pathlib import Path

def fix_file(filepath, fixes):
    """
    Apply fixes to a specific file
    
    Args:
        filepath: Path to the file
        fixes: List of tuples (line_number, fix_type, fix_function)
    """
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    modified = False
    for line_num, fix_type, fix_func in fixes:
        if line_num <= len(lines):
            old_line = lines[line_num - 1]
            new_line = fix_func(old_line)
            
            if old_line != new_line:
                lines[line_num - 1] = new_line
                print(f"  Line {line_num} ({fix_type}):")
                print(f"    Old: {old_line.strip()[:80]}...")
                print(f"    New: {new_line.strip()[:80]}...")
                modified = True
    
    if modified:
        with open(filepath, 'w') as f:
            f.writelines(lines)
        print(f"✅ Fixed {filepath}\n")
    else:
        print(f"ℹ️  No changes needed for {filepath}\n")
    
    return modified

def fix_whitespace_around_operator(line):
    """Fix E226: missing whitespace around arithmetic operator"""
    # Fix common operators without spaces
    patterns = [
        (r'(\w)\*(\w)', r'\1 * \2'),  # Fix multiplication
        (r'(\w)/(\w)', r'\1 / \2'),    # Fix division
        (r'(\w)\+(\w)', r'\1 + \2'),   # Fix addition
        (r'(\w)-(\w)', r'\1 - \2'),    # Fix subtraction
        (r'(\))\*(\d)', r'\1 * \2'),   # Fix ) * number
        (r'(\d)\*(\()', r'\1 * \2'),   # Fix number * (
    ]
    
    new_line = line
    for pattern, replacement in patterns:
        new_line = re.sub(pattern, replacement, new_line)
    return new_line

def fix_long_line(line, max_length=100):
    """Fix E501: line too long"""
    if len(line.rstrip()) <= max_length:
        return line
    
    # For different types of long lines, apply different strategies
    
    # If it's a string, try to break it
    if '"""' in line or "'''" in line:
        return line  # Don't break docstrings
    
    # If it's a function call with many parameters
    if '(' in line and ')' in line:
        # Find good breaking points (after commas)
        if ',' in line:
            parts = line.split(',')
            if len(parts) > 1:
                # Try to break after a comma near the middle
                indent = len(line) - len(line.lstrip())
                new_line = parts[0] + ',\n'
                for i, part in enumerate(parts[1:], 1):
                    if i == len(parts) - 1:
                        new_line += ' ' * (indent + 4) + part
                    else:
                        new_line += ' ' * (indent + 4) + part + ',\n'
                return new_line
    
    # If it's a long string literal
    if '"' in line or "'" in line:
        # Check if we can use parentheses to break the string
        match = re.search(r'(.*?)(["\']{1,3})(.*?)\2(.*)', line)
        if match and len(match.group(3)) > 50:
            # Break long strings using implicit concatenation
            indent = len(line) - len(line.lstrip())
            before = match.group(1)
            quote = match.group(2)
            content = match.group(3)
            after = match.group(4)
            
            # Break content into chunks
            chunk_size = 70
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            
            if len(chunks) > 1:
                new_line = before + '(\n'
                for chunk in chunks:
                    new_line += ' ' * (indent + 4) + quote + chunk + quote + '\n'
                new_line = new_line.rstrip() + '\n' + ' ' * indent + ')' + after
                return new_line
    
    # Default: try to break at logical operators or comparison operators
    for op in [' and ', ' or ', ' if ', ' else ']:
        if op in line:
            parts = line.split(op, 1)
            if len(parts) == 2:
                indent = len(line) - len(line.lstrip())
                return parts[0] + op + '\n' + ' ' * (indent + 4) + parts[1].lstrip()
    
    return line  # Return unchanged if no good breaking point found

def remove_unnecessary_fstring(line):
    """Fix F541: f-string is missing placeholders"""
    # Look for f-strings without any placeholders
    if 'f"' in line or "f'" in line:
        # Check if there are no curly braces (placeholders)
        if '{' not in line:
            # Remove the f prefix
            line = re.sub(r'\bf"', '"', line)
            line = re.sub(r"\bf'", "'", line)
    return line

def main():
    """Main function to fix all linting issues"""
    print("🔧 Fixing all linting issues in fraud-detection-system\n")
    print("="*60)
    
    # Define all fixes needed based on flake8 output
    fixes_map = {
        "src/data_processing/generate_data.py": [
            (131, "E226", fix_whitespace_around_operator),
        ],
        "src/models/fraud_detector.py": [
            (109, "E501", fix_long_line),
            (592, "F541", remove_unnecessary_fstring),
        ],
        "src/monitoring/dashboard.py": [
            (234, "E501", fix_long_line),
            (369, "E226", fix_whitespace_around_operator),
            (379, "E226", fix_whitespace_around_operator),
        ],
        "src/monitoring/model_monitor.py": [
            (569, "E226", fix_whitespace_around_operator),
            (569, "E501", fix_long_line),  # Same line has two issues
        ],
    }
    
    # Apply fixes to each file
    total_fixed = 0
    for filepath, fixes in fixes_map.items():
        print(f"\n📄 Processing: {filepath}")
        print("-" * 40)
        if fix_file(filepath, fixes):
            total_fixed += 1
    
    print("="*60)
    print(f"\n✨ Summary: Fixed {total_fixed} out of {len(fixes_map)} files")
    
    # Provide next steps
    print("\n📋 Next Steps:")
    print("1. Run the linting commands again to verify all issues are fixed:")
    print("   isort src tests")
    print("   black src tests")
    print("   flake8 src tests --max-line-length=100 --ignore=E203,W503")
    print("\n2. If any issues remain (especially long lines), they may need manual intervention")
    print("   as some lines might be difficult to break automatically while maintaining readability.")
    print("\n3. Review the changes with:")
    print("   git diff")
    print("\n4. If satisfied, commit the changes:")
    print("   git add -A")
    print('   git commit -m "Fix linting issues (E226, E501, F541)"')

if __name__ == "__main__":
    main()
