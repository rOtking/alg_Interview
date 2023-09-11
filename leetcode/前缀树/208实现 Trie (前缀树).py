class TrieNode:
    def __init__(self) -> None:
        self.passNum = 0
        self.end = 0
        self.nexts = [None] * 26


class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        if word is None or len(word) == 0:
            return
        node = self.root
        node.passNum += 1

        for ch in word:
            index = ord(ch) - ord('a')
            # 不存在
            if node.nexts[index] is None:
                node.nexts[index] = TrieNode()
            node = node.nexts[index]
            node.passNum += 1
        node.end += 1

        return

    def search(self, word: str) -> bool:
        if word is None or len(word) == 0:
            return False

        node = self.root
        for ch in word:
            index = ord(ch) - ord('a')
            if node.nexts[index] is None:
                return False
            node = node.nexts[index]

        return True if node.end > 0 else False

    def startsWith(self, prefix: str) -> bool:
        if prefix is None or len(prefix) == 0:
            return False
        node = self.root

        for ch in prefix:
            index = ord(ch) - ord('a')
            if node.nexts[index] is None:
                return False
            node = node.nexts[index]
        # 当然可以返回数量 passNum
        return True

# ok！