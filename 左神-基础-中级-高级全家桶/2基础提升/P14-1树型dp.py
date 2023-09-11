''''''
# ------------------------- 1.BT结点间最大距离问题 ------------------ #
'''
从某个a出发，可向上可向下，但每个结点只能走一次，到达b，路径上的结点个数为ab的距离。

求BT上的最大距离。


思路：考虑以x点为head的树最大距离，由子树的已知结果构成。常见分类就是结果是否包含x本身：
        1.不包括：
            （1）x的最大就是左子树的最大；
            （2）就是右子的最大；
        2.包括：
             左子的最远结点 + 1本身 + 右的最远（其实就是树的高度）
        
注意：一个递归过程是可以返回多种结果的！

很精彩！

实现很简单，重要的是结果是怎么构成的。
'''
def maxDistanceInTree(head):
    # 返回 最大距离与高度
    def process(head):
        if head is None:
            return 0, 0
        res1 = process(head.left)
        res2 = process(head.right)
        res3 = res1[1] + 1 + res2[1]

        maxDis = max(res1[0], res2[0], res3)
        maxHeight = max(res1[1], res2[1]) + 1

        return maxDis, maxHeight


    return process(head)[0]

# ---------------------------------------------------- #


# ------------------------- 2.树型DP套路 ------------------ #
'''
1.从左子树，右子树与整棵树的角度考虑可能性；
2.由可能性列出所有需要的信息；
3.合并信息，得到结果，返回信息结构；
4.basecase

'''
# ---------------------------------------------------- #
# ------------------------- 3.派对的最大快乐值 ------------------ #
class Employee:
    def __init__(self, happy, subordinates):
        self.happy = happy    # 这名员工能带来的快乐值
        self.subordinates = subordinates   # 直接下级
'''
整个公司人员可看为无环多叉树。head是老板，所有员工都有唯一上级。叶结点是基层员工无下级。办party，原则：
（1）某个员工来，他的所有直接下级不能来；
（2）party快乐值是所有人和；
（3）目标是让快乐最大。

给头结点boss，返回最大值。

            x 
          / | \           
         a  b  c   
思路：以x为头的集体最大值分为
（1）x参与:a不来时a整棵树的最大 + b不来时b整棵树的最大 + c不来时c整棵树的最大 

（2）x不参与:max(a来是整棵树最大,a不来整棵最大) + max(b来是整棵树最大,b不来整棵最大) + max(c来是整棵树最大,c不来整棵最大)

process可返回head来时的最大与不来时的最大两个res！！！

核心就是分析结果构成！
'''

def maxHappy(boss):
    def process(employee):
        # 叶子
        if len(employee.subordinates) == 0:
            return employee.happy
        # 参与
        res1 = 0
        # 不参与
        res2 = 0
        for subordinate in employee.subordinates:
            res1 += process(subordinate)[1]
            res2 += max(process(subordinate))

        return res1, res2

    return max(process(boss))



