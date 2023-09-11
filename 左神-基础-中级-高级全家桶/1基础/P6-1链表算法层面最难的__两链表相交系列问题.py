'''
链表算法层面最难系列：相交问题
'''
# --------------------- 1.链表有环相交 ---------------#
'''
问题：两单list，不知道有没有环，求第一个相交结点；不相交None。时间O(n),空间O(1)

1.求是不是有环？返回入环结点：

（1）如果不要求空间，可以用set的方式，每次查一下，第一个重复的就是入环节点；
（2）空间O(1)的做法：2*(a+c) = a+b+c理论；

2.head1与head2都无环：先走差值步，常规判断即可；
3.一个有环，一个无环：不可能相交！

4.都有环loop1,loop2
（1）若loop1==loop2：
      \ 
       \   /
        \ /
         |
     ____|___
    |        |
    |        |
    |___<____|
看作loop点之前的无环相交问题即可，相交点一定在loop点前；

（2）loop1在回到自己之前没遇到loop2，则不相交；         
      \            
       \   /
        \ /
         |
     ____|___
    |        |
    |        |
    |___<____|  

      \ 
       \   /
        \ /
         |
     ____|___
    |        |
    |        |
    |___<____|

（3）loop1在回到自己之前遇到loop2，交两个点，返回任意一个；
   \          /
    \________/
    |        |
    |        |
    |___<____| 
    
    
    
核心：多种情况分类讨论清楚即可！    不实现了！
'''