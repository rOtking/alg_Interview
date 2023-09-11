'''
'''

'''
KMP：求子串的问题。

如
str1:abc1234de,  str2:1234  问str2是不是str1的字串。  字串是连续的。

暴力解法：每个位置为开头，依次遍历，如果str1长度n，str2长度m，则时间为O(n*m)

如果str1:1111112   str2:1112    会有很多次没用的操作。


KMP就是加速这个过程。

-----------------------------------------------------------------
1.最长前缀后缀匹配长度：

        abbabb(?)
对该位置：
        长度  1  2   3       4     5        6
        前缀  a  ab  abb    abba  abbab    不考虑
        后缀  b  bb  abb    babb  bbabb
     是否相等  N  N   Y       N     N
     
不考虑前缀后缀一样的情况，最长长度为3。

每个位置得一个长度，构成的数组  nextArr，认为规定 0位置是-1，1位置是0

        位置:  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14
  如   str2:   a  a  b  a  a  b  s  a  a  b  a   a   b   s   t
    nextArr:  -1  0  1  0  1  2  3  0 ....
    
对str2求nextArr即可。

 
-----------------------------------------------------------------
2.KMP比对过程
 x是str1的指针，y是str2的

str1:   i ... x ...
str2:   0 ... y ...

str1从i位置开始与str2的0位置比，直到第一个不同出现，即str1的x位置与str2的y位置不同。
        暴力过程：x跳到str1的i+1位置，y跳到str2的0位置重新开始。
        即
                   x 
        str1:   i i+1 ...
        
        str2:      0  ... 
                   y 


KMP利用str2的nextArr信息，设nextArr[y] = k
str1:   i       ...                  x ...
str2:   0 ... k-1 k ... y-k  ... y-1 y ...
        |__k___|         |___k____|
    
    直接下一次用str2的k位置与str1的x位置比
    即
    str1:   i       ...       x-k        x ...
    str2:                      0 ... k-1 k ... y'-k  ... y'-1 y' ...
                                |__k___| ^       |___k____|
                                         y 
        
显然，相比暴力解，跳过了很多不可能的位置，完成了加速。

相当于y跳到k位置，与str1的x位置继续比。

（1）为什么不从x-k位置与str2的0位置开始比？因为没必要

    因为str1的 [x-k,x-1] 与 str2的[y-k,y-1]一样，str2的[y-k,y-1]又和自己的前缀[0.k-1]一样，所以
    str1的 [x-k,x-1]与str2的 [0.k-1]不用比了，肯定一样。从下个位置开始比就行了。
    就是把str2向右推，跳过一些不可能的解的过程。

（2）对于str1来说，i到x-k-1的任何位置的开头都一定配不成！原因很关键！核心！

    反证法。假设(i, x-k)上有个点p，以他为开头能配出str2
    
    str1:   i ... p ... x-k ... x
    str2:   0  k..L             y

设p到x-1距离为L，则str1的[p,x-1]一定与str2的[0,L]相等，且L>k;
又因为前提是str1的[i,x-1]与str2的[0,y-1]相等，所以str1的[p,x-1]与str2的[L,y-1]相等

所以得到str2的[0,L]与[L,y-1]相等L>k，矛盾！所以不可能！


完美举例：

str1: abbsabbtcabbsabbe...
str2: abbsabbtcabbsabbw...
（1）w的前缀为t
str1: abbsabb tc abbsabb e...
str2: abbsabb tc abbsabb w...    

则跳到t与e比较，相当于右推str2
str1: abbsabbtcabbsabbe ...
str2:          abbsabbt cabbsabbw...

（2）t != e  t跳到它前缀的下一个，s
str1: abbsabbtcabbsabbe ...
str2:              abbs abbtcabbsabbw...

（3）s != e   跳到0位置
str1: abbsabbtcabbsabbe ...
str2:                 a bbsabbtcabbsabbw...

（4）当str2是0位置时还不相等，str1跳下一个位置，重新比较
str1: abbsabbtcabbsabbe ...
str2:                   abbsabbtcabbsabbw...
  
'''
# todo 返回str1的起始位置
def kmp(str1, str2):
    if str1 is None or str2 is None or len(str2) < 1 or len(str1) < len(str2):
        return -1

    nextArr = getNextArr(str2)
    i = 0  # str1的指针
    j = 0  # str2的指针
    while(i < len(str1) and j < len(str2)):
        if str1[i] == str2[j]:
            i += 1
            j += 1
        # 首位置就不一样
        elif nextArr[j] == -1:
            i += 1
        # next前跳
        else:
            j = nextArr[j]
    # todo 有一个越界 j越界说明匹配了，i越界就没有
    return i - j if j == len(str2) else -1
'''
复杂度分析：

因为j会往回跳，所以不好分析。分析i与i-j
            i    i - j
    （1）  增大     不变
    （2）  增大     增大
    （3）  不变     增大
         0-->N    0-->N
while过程中这两个量都是不减小的！且每次直走一个分支，可见幅度最大就是2N，所以时间是O(N)级别。
'''

'''
------------------------------------------
加速求取nextArr:   利用i-1位置求i位置

    位置  0  1  ... i-1 i  
nextArr  -1 0       k

（1）如果str[k] == str[i-1]，则nextArr[i] = k + 1
不可能更长！还是反证法！
    
    如果nextArr[i]=p>k+1，即str[0..p-1] == str[i-p,i-1],即str[0...p-2]==str[i-p,i-2],
    即nextArr[i-1]==p-1>k，相悖！

（2）若str[k] != str[i-1]，看str[nextArr[k]] ？str[i-1]，不断套娃！直到找到，或为0。

举例！

位置                          i-1 i 
str      abbstabb ec abbstabb ?
nextArr     0     3           8

if str[i-1] == str[nextArr[i-1]]:
    (即 ? == str[8]，即? == e)
    nextArr[i] = nextArr[i - 1] + 1
else:
    if str[i-1] == str[nextArr[8]]:
        (即e的nextArr位置3,即 ? == s)
        nextArr[i] = nextArr[8] + 1 = 3 + 1
    else:
        if str[i-1] == str[nextArr[3]]:
            （即 ? == str[0]）
            nextArr[i] = 0 + 1
        else:  与开头都不相等
            nextArr[i] =0
            
中间为什么能跳过，还是反证。

位置   0 ...  k-1 k ...p-1...  i-1 i
next                           k
str[i-1]!=str[k],但是str[0...p-1] == str[i-p,i-1],p-1>k
        肯定有str[0...p-2] == str[i-p,i-2]  长度p-1>k与假设不符合！
所以可以跳！

'''
def getNextArr(s):
    if len(s) == 1:
        return [-1]
    # 人为规定
    nextArr = [-1, 0]

    i = 2
    j = 0   # todo 用哪个位置与i-1比，也是nextArr的信息！双重含义
    while(i < len(s)):
        if s[i-1] == s[j]:
            i += 1
            j += 1   # todo 确保nextArr[i+1]的值正确
            nextArr.append(j)
        # todo 不等，j向前跳
        elif j > 0:
            j = nextArr[j]
        # j是str的0位置，不能跳了
        else:
            i += 1
            nextArr.append(0)

    return nextArr

'''
复杂度分析：

M = len(str2)
因为j会往回跳，所以不好分析。分析i与i-j
            i    i - j
    （1）  增大     不变
    （2）  不变     增大  
    （3）  增大     增大
         0-->M    0-->M
while过程中这两个量都是不减小的！且每次直走一个分支，可见幅度最大就是2N，所以时间是O(M)级别。



整体复杂度：
N>M，所以整体线性O(N)
'''


