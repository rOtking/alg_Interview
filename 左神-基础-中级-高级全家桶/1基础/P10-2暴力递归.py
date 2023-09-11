'''

'''
'''
暴力递归的核心就是尝试，也是最难的的，天赋 + 刷的多有经验：
（1）原问题转为缩小规模的同类子问题；
（2）明确的结束条件 base case；
（3）得到子问题的结果进行决策，解决原问题；
（4）先不记录子问题的解，那是dp优化加速做的事。
'''


# ---------------------- 1.汉诺塔问题 ----------------------#
'''
ABC3个柱子，n是A初始的盘子个数，大的必在小的下面。每次移动一个，最少步数从A到C

定义函数process(n, from, to, other)为from上的前n个盘子通过other移动到to上去：
则求 process(n, A, C, B)即可：

（1）把A的前n-1移到B上；
（2）把A的n移动到C；
（3）把B移动到C上，完成。


'''
def hanoi(n):
    def process(n, source, target, other):
        if n < 1:
            return


        process(n - 1, source, other, target)
        # process(1, source, target, other)
        # todo 直接放，不需要经过other
        print('from ' + source[0] + ' to ' + target[0])
        target.append(source.pop())
        print(source)
        print(other)
        print(target)
        print('----------------------------')

        process(n - 1, other, target, source)


    source = ["A"]
    for _ in range(n - 1, -1, -1):
        source.append(str(_))
    process(n, source, ['C'], ['B'])

# hanoi(3)
# todo ok 定义好process的含义！考虑大的调度！
# --------------------------------------------------------- #
# ---------------------- 2.打印字符串的全部子序列 ----------------------#
'''

就是从0开始的每个位置，都可以选择要或不要。

所以可以根据每个结点要不要把所有可能性构建成一个满二叉树。
'''
def allSubquences(chs):

    res = []
    if len(chs) == 1:
        res.append(chs)
        res.append('')
        return res

    subReses = allSubquences(chs[1:])
    for sub in subReses:
        res.append(chs[0] + sub)
        res.append(sub)
    return res

# subs = allSubquences('abc')
# print(subs)
'''
['abc', 'bc', 'ac', 'c', 'ab', 'b', 'a', '']
'''

# --------------------------------------------------------- #
# ---------------------- 2.打印字符串的全排列 ----------------------#
'''
1.全排列    n*(n-1)*...*1 每个位置的可能性
2.不重复的全排列。


试法：每个位置试！
'''
# --------------------------------------------------------- #
import copy
def allSort(chs):
    # 返回全排列
    def process(chsList):
        res = []
        if len(chsList) == 1:
            res.append(chsList)
            return res
        for ch in chsList:
            newList = copy.deepcopy(chsList)
            newList.remove(ch)
            tmpRes = process(newList)
            for ele in tmpRes:
                ele.insert(0, ch)
            res.extend(tmpRes)

        return res
    chsList = list(chs)
    tmp = process(chsList)
    res = []
    for ele in tmp:
        res.append(''.join(ele))

    print(res)
    return res

# todo ok了！注意返回的结构是什么！一直都是List[List[str]]
# allSort('abcd')

'''
感觉是个dfs的应用啊？试试！
'''
def allSortDFS(chs):
    records = []
    def dfs(track, n):
        # todo allAandidates始终是所有选择，使用更方便一些
        # track是做过的选择 放的是char在allAandidates的index，好区分重复的char

        # 终止条件
        if len(track) == n:
            records.append(track[:])
            return
        # 筛选候选人
        newCandidateIndexs = []
        for index in range(n):
            if index not in track:
                newCandidateIndexs.append(index)

        # 做选择
        for index in newCandidateIndexs:
            track.append(index)
            dfs(track, n)
            track.pop()
        return

    chsList = list(chs)
    dfs([], len(chs))
    print(records)


'''
实质：从dfs的参数也能看出来，就是对位置进行全排序，与是什么字符没关系的！
    所以，进一步，需要去掉重复的，如'aaa'按上面排列，会出现6个'aaa'！
    那就需要用到字符本身了，而不是单纯的对位置进行排序。
    
    应该也简单，写个注册表呗，注册过在同一级别就不再试了。
    
其实BFS应该也行吧？
        BFS善于遍历每个点，进行对应的处理；而DFS天然的把路径放到track里了。
        BFS也能存个track，比较复杂，不尝试了。
'''
def allSortDFSPlus(chs):
    records = []
    def dfs(track, n, allCandidates):
        # todo allCandidates始终是所有选择，使用更方便一些
        # track是做过的选择 放的是char在allAandidates的index，好区分重复的char

        # 终止条件
        if len(track) == n:
            records.append(track[:])
            return
        # 筛选候选人
        newCandidateIndexs = []

        # todo 就是这里，注册表
        visited = set()
        for index in range(n):
            if index not in track and allCandidates[index] not in visited:
                newCandidateIndexs.append(index)
                visited.add(allCandidates[index])

        # 做选择
        for index in newCandidateIndexs:
            track.append(index)
            dfs(track, n, allCandidates)
            track.pop()
        return

    chsList = list(chs)
    dfs([], len(chs), chsList)
    print(records)

# allSortDFS('aaa')
# allSortDFSPlus('aaa')
'''
[[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
[[0, 1, 2]]

结果正确！
'''

# ---------------------------------------------- #

# ---------------------- 3.取牌博弈 ----------------------#
'''
arr，两人取数，每次只能从首尾取。两个都绝顶聪明，每一步都让自己最终最大。求胜者的值。

arr = [1,2,100,4]
甲不会拿4，因为这样乙就能拿100，直接赢了。

arr = [1,100,2]   甲怎么都输。


可见，一开始的先后，就已经决定了输赢！


试法：在一段范围上试！

显然定义先后手最大值函数。
'''
def getMax(arr):
    # l,r上先手
    def f1(arr, l, r):
        if l == r:
            return arr[l]
        # todo 很巧妙！
        return max(arr[l] + f2(arr, l + 1, r), arr[r] + f2(arr, l, r - 1))


    # 后手
    def f2(arr, l, r):
        if l == r:
            return 0

        # todo 让接下来别人的先手最大值 最小
        return min(f1(arr, l + 1, r), f1(arr, l, r - 1))

    return max(f1(arr, 0, len(arr) - 1), f2(arr, 0, len(arr) - 1))

# ok
# res = getMax([1,2,100,4])
# print(res)

# ---------------------------------------------- #

# ---------------------- 4.逆序栈 ----------------------#
'''
只能用递归，不能用额外的数据结构。


待定吧，没啥意思。
'''

# ---------------------------------------------- #
# ---------------------- 5.转化结果 ----------------------#
'''
1--A，2--B，...26--Z。0不能转
111:AK或AAA或KA
给个str，求有多少种转化。

f()
就是递归。每个位置试：
（1）当前1，两种；（2）当前2，看后一位；（3）当前3-9，1种
'''
# chs从第i为之后的部分，能转为多少种
def getTransNum(chs, i):
    if '0' in chs:
        return 0
    if i == len(chs) - 1:
        return 1

    # i及后面至少2位数
    num = 0
    if chs[i] == '1':
        res = getTransNum(chs, i + 1)
        if i + 2 <= len(chs) - 1:
            res += getTransNum(chs, i + 2)
        else:
            res += 1
        return res

    elif chs[i] == '2':
        res = getTransNum(chs, i + 1)
        if int(chs[i + 1]) <= 6:
            if i + 2 <= len(chs) - 1:
                res += getTransNum(chs, i + 2)
            else:
                res += 1
        else:
            pass
    else:
        res = getTransNum(chs, i + 1)

    return res


# chs = getTransNum('2120', 0)
# print(chs)


# ---------------------------------------------- #
# ---------------------- 6.最大重量 ----------------------#
'''
weight[i]对用value[i]，bag最大重量的袋子，能装最多的价值。

试法：每个位置要或不要挨个试！
'''

def maxValue(weights, values, i, bag):
    '''
    从i开始，能获取的最大价值
    :param weights: 
    :param values: 
    :param i: 
    :param bag: 重量限制
    :return: 
    '''
    if i > len(weights) - 1 or bag <=0:
        return 0

    res1 = 0
    if weights[i] <= bag:
        res1 = values[i] + maxValue(weights, values, i + 1, bag - weights[i])

    res2 = maxValue(weights, values, i + 1, bag)
    return max(res1, res2)


# res = maxValue(weights=[1,2,3], values=[30,20,10], i=0, bag=2)
# print(res)
'''
不确定对不对，但是试了几个例子，没啥问题。
'''
# ---------------------------------------------- #
