''''''
# ----------------------- 1.Morris遍历 --------------------- #
'''
不论递归还是非递归的stack遍历，空间都是O(高度)

Morris遍历是时间O(N),空间O(1)。可在此基础上改出前序、中序与后序。

面试印象分upup！



1.细节：
当前结点cur，cur从head结点开始
(1)若cur没left，cur = cur.right
(2)若有left，找到左子树的最右结点mostRight
    <1>如果mostRight的right为空，让其指向cur，cur左移动, cur = cur.left
    <2>如果mostRight的right指向cur，让其指向None，cur右移， cur = cur.right
(3)cur为None是停止。


原理：不用stack怎么回到上级？  利用叶结点的空闲指针！当然如果严格规定不能修改BT，那Morris就不行。

有左结点的会遍历两次。是可以区分当前是第几次来的，cur的左子树的mostRight的right如果指向NOne，那就是第一次到cur；
                    如果是cur自己，那就是第二次，将它还原。
                    
                    
                    例子
                    
                    cur 1 
                /           \ 
               2             3
            /    \         /    \ 
           4      5       6      7
          / \    / \     / \    / \ 
         N  N    N N     N  N  N   N
遍历：1

（1）cur在1，有left，mostRight是5,5的right为N，指向1，cur到2
                      1 
                /     ^     \ 
          cur  2      |       3
            /    \    |     /    \ 
           4      5   |    6      7
          / \    / \__|  / \     / \ 
         N  N    N       N  N   N   N
遍历：1 2
（2）cur在2，有left，mostRight是4，4的r是N，指向2，cur左移动到4
                      1 
                /     ^     \ 
              2       |      3
            / ^   \   |     /   \ 
      cur  4  |    5  |    6      7
          / \_|   / \_|   / \    / \ 
         N       N       N  N   N   N
遍历：1 2 4
（3）cur在4，无left，cur到right，2
                      1 
                /     ^     \ 
         cur  2       |      3
            / ^   \   |     /   \ 
           4  |    5  |    6      7
          / \_|   / \_|   / \    / \ 
         N       N       N  N   N   N
遍历：1 2 4 2

（4）cur在2，左的mostRight是4，有right，是cur，则4的right还原None，cur右移到5
                      1 
                /     ^     \ 
              2       |      3
            /     \   |     /   \ 
           4   cur 5  |    6      7
          / \     / \_|   / \    / \ 
         N   N    N       N  N   N   N
遍历：1 2 4 2 5

（5）cur=5，无left，cur到right，cur=1
                  cur 1 
                /     ^     \ 
              2       |      3
            /     \   |     /   \ 
           4       5  |    6      7
          / \     / \_|   / \    / \ 
         N   N    N       N  N   N   N
遍历：1 2 4 2 5 1

（6）cur=1，有left。左的mostRight 5的right为cur，还原为None，cur右移到3
                     1 
                /          \ 
              2         cur 3
            /     \        /   \ 
           4       5      6      7
          / \     / \    / \    / \ 
         N   N    N N    N  N   N   N
遍历：1 2 4 2 5 1 3

（7）cur=3，有left，mosRight为6，6的right为N，指向cur，cur左移到6
                     1 
                /          \ 
              2             3
            /     \        /^  \ 
           4       5  cur 6 |    7
          / \     / \    / \|   / \ 
         N   N    N N    N     N   N
遍历：1 2 4 2 5 1 3 6

（8）cur=6，无left，cur到右，cur=3
                     1 
                /          \ 
              2         cur 3
            /     \        /^  \ 
           4       5      6 |    7
          / \     / \    / \|   / \ 
         N   N    N N    N     N   N
遍历：1 2 4 2 5 1 3 6 3

（9）cur=3，左的mostRight的right为cur，还原NOne，cur右移到7
                     1 
                /          \ 
              2             3
            /     \        /  \ 
           4       5      6 cur7
          / \     / \    / \   / \ 
         N   N    N N    N  N  N  N
遍历：1 2 4 2 5 1 3 6 3 7

（10）cur=7，无left，cur右移到None，结束。



2.对比实质

递归：每个结点都会到3次，有递归栈保存信息

Morris是模拟这个过程：（1）有左子树的结点会到两次，先去左子树走一遍，再回来；再去右子树。
                   （2）没有左子树的，直接去右子树。
            只要向右走了，就不会回来了，所以不会有第三次回来！
'''

def morris(head):
    if head is None:
        return

    cur = head

    mostRight = None

    while(cur is not None):
        mostRight = cur.left

        # 如果有左孩子
        if mostRight is not None:
            # 找左树最右
            # todo 因为右孩子可能会修改，要找他修改前的位置才是mostRight
            while(mostRight.right is not None and mostRight.right is not cur):
                mostRight = mostRight.right
            # 此时找到最右了

            # 第一次到cur
            if mostRight.right is not cur:
                mostRight.right = cur
                cur = cur.left
                continue   # 大过程继续
            else:
                mostRight.right = None   # 还原 后面统一右移

        # 不论有没有左，最终都向右
        cur = cur.right

'''
               3. 复杂度分析
                
空间明显O(1)，就cur与mostRight两个。

所有结点遍历左子树的右边界两遍，其实就是沿着每条右边界画斜线的结点遍历两次，每条斜线结点都是不重复的！所以一共就是O(2N)也就是O(N)


'''
'''
            1 2 4 2 5 1 3 6 3 7
        次数 1 1 1 2 1 2 1 1 2 1
4.改前序：
        如果一个node只能到1次，直接打印；
        可到两次，第一次打印。

        1 2 4 5 3 6 7

代码两点修改！
'''
def morrisPre(head):
    if head is None:
        return
    cur = head
    mostRight = None
    while(cur is not None):
        mostRight = cur.left

        if mostRight is not None:
            while(mostRight.right is not None and mostRight.right is not cur):
                mostRight = mostRight.right

            # 第一次到cur
            if mostRight.right is not cur:
                # todo 修改点1
                print(cur.value)
                mostRight.right = cur
                cur = cur.left
                continue   # 大过程继续
            else:
                mostRight.right = None   # 还原 后面统一右移

        # todo 修改点2
        else:
            print(cur.value)

        # 不论有没有左，最终都向右
        cur = cur.right


'''
            1 2 4 2 5 1 3 6 3 7
        次数 1 1 1 2 1 2 1 1 2 1
5.改中序：
        如果一个node只能到1次，直接打印；
        可到两次，第2次打印。

        4 2 5 1 6 3 7

代码一点点修改！
'''
def morrisIn(head):
    if head is None:
        return

    cur = head
    mostRight = None
    while(cur is not None):
        mostRight = cur.left
        if mostRight is not None:
            while(mostRight.right is not None and mostRight.right is not cur):
                mostRight = mostRight.right

            # 第一次到cur
            if mostRight.right is not cur:
                mostRight.right = cur
                cur = cur.left
                continue   # 大过程继续
            else:
                mostRight.right = None   # 还原 后面统一右移
        # todo 修改点 有没有左树，都从这出去！
        print(cur.value)
        # 不论有没有左，最终都向右
        cur = cur.right


'''
            1 2 4 2 5 1 3 6 3 7
        次数 1 1 1 2 1 2 1 1 2 1
6.改后序：
        只在能两次到的node且第二次到的时候打印，逆序打印它左子树的右边界。
        结束后，单独逆序打印整棵树的右边界。

            1
           / \ 
          2   3
         / \ / \ 
        4  5 6  7

（1）第二次到2时打印：4
（2）第二次到1时打印：5，2
（3）第二次到3时打印：6
（4）最后，整棵树右边界：7，3

如何有限遍历逆序打印左树右边界？----- 把 右边界的right看成list的next，先链表逆序，打印后，再逆序回来。

两个修改点
'''
def morrisPost(head):
    if head is None:
        return
    cur = head
    mostRight = None

    while(cur is not None):
        mostRight = cur.left

        if mostRight is not None:
            while(mostRight.right is not None and mostRight.right is not cur):
                mostRight = mostRight.right
            # 第一次到cur
            if mostRight.right is not cur:
                mostRight.right = cur
                cur = cur.left
                continue   # 大过程继续
            else:
                mostRight.right = None   # 还原 后面统一右移
                # todo 修改点1  第二次到达时逆序打印左树右边界
                printRightEdge(cur.left)

        # 不论有没有左，最终都向右
        cur = cur.right
    # todo 修改点2  整棵树右边界
    printRightEdge(head)


# 逆序
def reverseEdge(x):
    if x is None:
        return x
    pre = None
    cur = x

    while(cur):
        right = cur.right
        cur.right = pre

        pre = cur
        cur = right

    return pre


# 逆序打印x为head的右边界
def printRightEdge(x):
    head = reverseEdge(x)
    cur = head
    while(cur):
        print(cur.value)
        cur = cur.right

    reverseEdge(head)

# ----------------------- 2.Morris应用 --------------------- #
'''
判断一树是不是BST。

中序升序。单独存下来的空间大。用Morris中序简单修改。

Morris解决本质的遍历问题，是BT遍历的最优解。

'''
def isBST(head):
    if head is None:
        return True

    cur = head
    mostRight = None
    preValue = float('-inf')
    while(cur is not None):
        mostRight = cur.left
        if mostRight is not None:
            while(mostRight.right is not None and mostRight.right is not cur):
                mostRight = mostRight.right

            # 第一次到cur
            if mostRight.right is not cur:
                mostRight.right = cur
                cur = cur.left
                continue   # 大过程继续
            else:
                mostRight.right = None   # 还原 后面统一右移
        # todo 比较
        if cur.value <= preValue:
            return False
        preValue = cur.value

        cur = cur.right

    return True


'''
                        总结
    
                    最优解什么时候用递归套路，什么时候Morris？

    1.必须做第三次遍历的强整合，用递归。
            如 先左树，再右树，最后第三次需要回到当前结点整合信息。

    2.不必第三次，Morris。





'''