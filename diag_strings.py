src = open("test_iframe_bitstream.py", encoding="utf-8").read()
cut = src.index("# Calculer la taille")
head = src[:cut]

tail = '''
MK = b"\\x42\\xE5\\x34\\x92"

def walk(o, path="strings"):
    if isinstance(o, (bytes, bytearray)):
        n = len(o)
        print(path, "bytes len =", n, "| len%4 =", n % 4, "| marqueur =", MK in o)
        if MK in o:
            s, m = o.split(MK)
            print("   flux_ctx len =", len(s), "| len%4 =", len(s) % 4,
                  "| minmax (octet) =", m[0], "| octets apres marqueur =", len(m))
    elif isinstance(o, (list, tuple)):
        print(path, type(o).__name__, "len =", len(o))
        for i, x in enumerate(o):
            walk(x, path + "[" + str(i) + "]")
    else:
        print(path, type(o))

print()
walk(strings)
print("shapes =", shapes)
'''

exec(compile(head + tail, "diag_run", "exec"))