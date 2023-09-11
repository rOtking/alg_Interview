'''
即字典树
Trie
'''
'''
是个N叉树，边上放的是a-z字符，结点放从root到本节点所有边上的字符组成的字符串。

公共前缀都在本子树上。

root是'' 空串。

作用：高效的存储与搜索 字符串 集合 的问题。
'''

class TrieNode:
    def __init__(self):
        self.passPath = 0  # 通过这个点的路径数
        self.end = 0      # 以这个结点为结束的个数

        self.nexts = [None] * 26   # todo 每个位置表示有没有向a-z的路径；缺点是没有的路径也占空间，空间浪费。

        # todo nexts可以用map存，哪个路径对应到哪个结点
        # self.nexts = {}     # k:char    v:TrieNode

'''
['abc', 'ab', 'bc', 'bck']

                o  p=4,e=0
             a/ | \ b  
    p=2,e=0  o      o  p=2,e=0
            b|      |c
    p=2,e=1  o      o  p=2,e=1
            c|      |k
    p=1,e=1  o      o  p=1,e=1
    

流程：
（1）初始结点，p=0,e=0

for 遍历每个str：

（2）对'abc'，遍历char,
    <1>先来到root结点,p=1；
    <2>当前结点有没有挂a，没有就创建新结点，直接p=1,e=0；
    <2>当前没挂b，创建,p=1，e=0;
    <3>当前没c，创建，p=1，e=0；
    <4>'abc'结束了，最后结点 e += 1
（3）对'ab'
    <1>先到root,p += 1;
    <2>当前有a，复用， p+=1；
    <3>当前有b，复用, p+=1;
    <4>b结束，e +=1
（4）'bc'
    <1>root的p += 1
    <2>root没直接挂b，创建，p=1，e=0
    <3>当前没c，创建，p=1，e=0；
    <4>结束，e+=1
（5）'bck'
    <1>root p += 1
    <2>root有b，p+=1；
    <3>当前有c，p+=1
    <4>当前c没k，创建，p=1，e=0；
    <5>结束，k的e+=1。
    

用途举例：

1.查之前有没有加入过某个str，如'bc'：
    直接按root-b-c去找即可，如果有路径 and c的e>0就是加入过。
    时间就是 O(字符个数)

map可完成上面👆，不能完成下面👇

2.查加入的str有多少是以'ab'为前缀的？
    找到root-a-b的路径，如果存在，b的p值就是。
    
    
    
其他：删除操作！先serch，存在再删除！

流程：
（1）root的p -= 1
for遍历每个ch
（2）每到一个ch，他的p -= 1
（3）最后一个结点的e-=1

如果过程中某个结点的p=0了，那该结点及之后路径结点都可以删除了！
'''
# todo 经验证 ok的！！！


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

    def delete(self, word):
        if self.search(word):
            node = self.root
            node.passNum -= 1
            for ch in word:
                index = ord(ch) - ord('a')
                node.nexts[index].passNum -= 1
                if node.nexts[index].passNum == 0:
                    del node.nexts[index]
                    node.nexts[index] = None
                    # python会自动析构,连的路径也就没了
                    return
                # 还存在
                node = node.nexts[index]
            node.end -= 1




