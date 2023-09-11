'''

'''
# -------------------------- 1.哈希函数 -------------------------- #
'''
1.经典hash函数f，特性：

（1）
输入域无穷，及无穷多的可能性，如输入任意字符串；

输出域有限，如md5范围 (0 , 2^64 - 1)，就是输入一个str，得到一个16bit的字符串，每位0-f，也就是16*16的范围，正好2^64

（2）相同输入一定得到相同输出-----及hash没有随机成分；

（3）不同输入有相同输出-----及哈希碰撞；但概率极低。

（4）均匀离散性：输入经过hash得到的输出在输出域上均匀离散，很接近的输入经过hash差别也很大变得无规律、输出域上任意等大小的区域内输出的数量
                几乎一样！    最重要的性质！


hash的实现有很多，但要维持的性质都是上面4个！均匀离散做的越好，hash越优秀！

2.进一步加工，
in --hash-> out --%m->0~m-1

输入经过hash在S域上均匀分布，输出在进行 %m运算的结果，也能保证在 0～m-1的范围上均匀分布。



'''
# todo
'''

应用1：一大文件，存了40亿数，无符号32位，只给你1G内存，统计出现最多的数。

传统的map统计内存一定爆。------>hash

for遍历每个数：
    hash(输入) % 100 ——————>>结果在0-99
    是几就放到几号文件，外存中。

把小文件加载进内存，统计每个小文件的次数，释放掉。依次统计，得到结果。

原因：40亿被均分到100个小文件，缩小规模。且同一个数一定在同一个文件内，好统计。

'''

# ----------------------------------------- #
# -------------------------- 2.哈希表的实现 -------------------------- #
'''
先开辟一段空间，假设n=17
[0 |
 1 |->Node(k='abc',v=10)
 2 |
...|
16 |
]
（1）每一个数x,hash(x)%m得到一个数，如k='abc',v=10,  hash('abc')%17=1
    则k='abc',v=10组成一个Node串在1号位置的链表上。
（2）因为hash性质，每个链表都均匀增长；
（3）get时就是按上述方法找到位置后，遍历list；
（4）给个thredshold=6，当某个list达到6长度，认为get域put都代价比较高，达不到O(1)，触发扩容；
（5）开辟一个n=34的空间，把原来的移动过去，每个list长度缩小一半；

如果threshold比较小，那就认为逼近O(1),扩容可以离线完成的！

'''

# ----------------------------------------- #
# -------------------------- 3.RandomPool -------------------------- #
'''
设计结构，
（1）insert(key)，加入key；（2）delete(key)；（3）getRandom()等概率随机获得一个key
每个时间O(1)

利用hash表就行，不是设计hash表。可用random函数

难点：random在连续区间获取，所以delete时与最后一个先交换，再删除。

通过额外维护一个index2key来实现随机。

'''
import random

class RandomPool:
    def __init__(self):
        self.size = 0
        self.key2index = {}
        self.index2key = {}

    def insert(self, key):
        self.key2index[key] = self.size
        self.index2key[self.size] = key
        self.size += 1

    def delete(self, key):
        if key not in self.key2index:
            return

        removeIndex = self.key2index[key]
        # 找最后一个
        self.size -= 1
        LastKey = self.index2key[self.size]

        # 交换
        self.key2index[LastKey] = removeIndex
        self.index2key[removeIndex] = LastKey

        self.key2index.pop(key)


    def getRandom(self):

        # 0-size上的随机index
        index = int(random.random() * self.size)
        return self.index2key[index]


# ----------------------------------------- #
