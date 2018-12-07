def title(title):
    s = f"# --- {title.upper()} TESTS "
    s = s + "-" * (78 - len(s)) + " #"
    print(s, flush=True)


def subtest(n, desc):
    print(f"# ({n}) {desc}: ", end='', flush=True)


def evaluate(test):
    print(end='', flush=True)
    print("PASS" if test == True else "FAIL", flush=True)


def end_tests():
    print("# " + "-" * 76 + " #", flush=True)
    print(flush=True)
