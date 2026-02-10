import sys
import os
import re

def process_line(line, level):
    # Match global_load* or global_store* instructions
    instr_pattern = r'\b(global_load\w*|global_store\w*)\b'
    instr_pattern_s = r'\b(s_load\w*)\b'
    if re.search(instr_pattern, line):
        # Skip if glc/dlc/sc0/sc1 is already present
        if re.search(r'\b(glc|dlc|sc0|sc1)\b', line):
            return line
        # If there is a comment, insert glc before it
        if ';' in line:
            parts = line.split(';', 1)
            code = parts[0].rstrip()
            comment = ';' + parts[1]
            return f"{code} sc1 sc0 nt {comment}"
        else:
            # No comment, just add glc at the end
            if level == "l1":
                return line.rstrip() + ' sc0 nt\n'
            elif level == "nt":
                return line.rstrip() + ' nt\n'
            elif level == "l2":
                return line.rstrip() + ' sc1 nt\n'
            elif level == "all":
                return line.rstrip() + ' sc0 nt sc1\n'
            else:
                return line.rstrip() + ' sc0 nt sc1\n'

    if True: #re.search(instr_pattern_s, line):
        # Skip if glc/dlc/sc0/sc1 is already present
        if re.search(r'\b(glc|dlc|sc0|sc1)\b', line):
            return line
        # If there is a comment, insert glc before it
        if ';' in line:
            parts = line.split(';', 1)
            code = parts[0].rstrip()
            comment = ';' + parts[1]
            return f"{code} glc {comment}"
        else:
            # No comment, just add glc at the end
            return line.rstrip() + ' glc\n'
    return line

def process_file(infile, level, out_file):
    with open(infile, 'r') as f:
        lines = f.readlines()
    new_lines = [process_line(line, level) for line in lines]
    with open(out_file, 'w') as f:
        f.writelines(new_lines)
    print(f'Generated: {out_file}')

def main():
    level    = sys.argv[2]
    asm_file = sys.argv[1]
    out_file = sys.argv[3]
    process_file(asm_file, level, out_file)

if __name__ == '__main__':
    main()
