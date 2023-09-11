'''
'
'''
'''
完成工作的顺序问题：如想完成A，依赖BCD,B又依赖CDE，问先后顺序。

就是有向图问题，不能有环，因为循环依赖不可以。有入度为0的结点。
            
            
            _______________
           |               |
           v               | 
   E ----> B ----> A <-----C
           ^       ^
           |       |
           |       |
           |______ D
                   
顺序：ECDBA   才能保证工作完成。


方法：
（1）找入度为0的点，E；  去掉它和它的影响
（2）新图中找入度为0的CD,去掉C
...

依次去掉即可。

代码不实现了，Graph的class得建立。
'''