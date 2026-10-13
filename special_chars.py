import regex



SPECIAL_CHARACTERS = regex.compile(r'[\p{P}\p{S}\p{Z}]')




# import re
# import unicodedata

# punctuation_chars = []
# symbol_chars = []
# space_chars = []

# for codepoint in range(0x110000):
#     char = chr(codepoint)
#     category = unicodedata.category(char)

#     if category.startswith("P"):  #Punctuation 标点
#         punctuation_chars.append(char)
#     elif category.startswith("S"):#Symbol 符号
#         symbol_chars.append(char)
#     elif category.startswith("Z"):  # 分隔符
#          space_chars.append(char)

# special_chars = punctuation_chars + symbol_chars + space_chars

# # 将特殊字符列表转换成正则表达式
# SPECIAL_CHARACTERS = re.compile(
#     "[" + re.escape("".join(special_chars)) + "]"
# )