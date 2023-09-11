class TrieNode:
    def __init__(self) -> None:
        self.passNum = 0
        self.end = 0
        self.nexts = [None] * 26


class TrieTree:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word):
        if word is None or len(word) == 0:
            return

        node = self.root
        node.passNum += 1

        for ch in word:
            index = ord(ch) - ord('a')
            if node.nexts[index] is None:
                node.nexts[index] = TrieNode()
            node = node.nexts[index]
            node.passNum += 1
        node.end += 1
        return

    # 查询返回word的最短前缀
    def shortPrefix(self, word):
        pre = ''
        if len(word) == 0:
            return pre

        node = self.root
        for ch in word:
            index = ord(ch) - ord('a')
            if node.nexts[index] is None:
                return word
            node = node.nexts[index]
            pre += ch
            if node.end > 0:
                return pre

        return word


class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        if sentence is None:
            return ''
        # 构建前缀树
        trieTree = TrieTree()
        for an in dictionary:
            trieTree.insert(an)
        sentences = sentence.split(' ')
        res = ''
        for s in sentences:
            res += (trieTree.shortPrefix(s) + ' ')
        return res[:-1]



# ok
# todo 典型的前缀树的应用！ 开心！





