class TrieNode:
    def __init__(self) -> None:
        self.passNum = 0
        self.end = 0
        self.value = 0
        self.nexts = {}


class MapSum:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, key: str, val: int) -> None:
        if key is None or len(key) == 0:
            return

        node = self.root
        node.passNum += 1
        for ch in key:
            if ch not in node.nexts:
                node.nexts[ch] = TrieNode()
            node.nexts[ch].passNum += 1
            node = node.nexts[ch]
        node.end += 1
        node.value = val

    def sum(self, prefix: str) -> int:
        if prefix is None or len(prefix) == 0:
            return 0

        node = self.root
        for ch in prefix:
            if ch not in node.nexts:
                return 0
            node = node.nexts[ch]
        return self.sumNode(node)

    # 计算以某个节点出发的和
    def sumNode(self, node):
        sumVal = node.value
        if len(node.nexts) == 0:
            return sumVal

        for k, v in node.nexts.items():
            sumVal += self.sumNode(v)

        return sumVal



# ok 前缀树的思想即可。