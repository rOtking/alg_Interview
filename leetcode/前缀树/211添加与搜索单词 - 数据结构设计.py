class TrieNode:
    def __init__(self) -> None:
        self.passNum = 0
        self.end = 0
        self.nexts = [None] * 26


class WordDictionary:

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        if word is None or len(word) == 0:
            return
        node = self.root
        node.passNum += 1
        for ch in word:
            index = ord(ch) - ord('a')
            if not node.nexts[index]:
                node.nexts[index] = TrieNode()
            node = node.nexts[index]
            node.passNum += 1
        node.end += 1

    def search(self, word: str) -> bool:
        # 如果是  . 从当前结点开始dfs 直到结束
        return self.dfs(self.root, word)

    def dfs(self, root, word):
        # 再以root的结点搜索word
        # 终止条件： word搜素结束，有以当前结点为结束的记录，那就存在；没找到就不存在
        if len(word) == 0:
            if root.end != 0:
                return True
            else:
                return False

        # 根据当前word[0]的状况构建新的candidates
        ch = word[0]
        index = None
        if ch != '.':
            index = ord(ch) - ord('a')
        # 当前是'.'：dfs找每个记录过的，有一条成功就结束；都不成功说明不存在
        if index is None:
            for node in root.nexts:
                if node:
                    res = self.dfs(node, word[1:])
                    if res:
                        return True
            return False
        # 当前是 a-z：有对应就看后面；没有直接False
        else:
            if root.nexts[index]:
                return self.dfs(root.nexts[index], word[1:])
            else:
                return False


# todo 还是挺难的。主要是在这个结构下的dfs不熟练，终止条件怎么设置。

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)