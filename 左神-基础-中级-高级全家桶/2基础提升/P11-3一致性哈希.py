'''
数据服务器的组织
'''
'''
业务端的逻辑服务器是一个个独立的，想加多少加多少，互相不影响。

数据服务器，如mysql是单机也不会有什么问题。但是数据如果很多，需要多台服务器进行分布式存储。

1.传统做法，假如m=3台服务器，hash(key)%3后，能把key均匀的分到3台上。但是如果加一台服务器，那全量数据都要重新计算hash值，分配服务器。

一致性哈希：保证数据迁移代价不高。

假设用md5，把hash值返回域想为一个环，0 .... 2^64-1, 0
若有3台机器，找他们唯一的标识如hostUrl或Mac地址，进行md5得一个数，3个值会划分这个环。

        -----m1------m2------m3--|
       |_________________________|

把3个hash值排序，[m1,m2,m3]，待查询的key找顺时针最近的结点作为存储的归属。如key在m1-m2，那就属于m2。

好处：如果想增加一个m4机器，算出来他在m1与m2之间，

        -----m1--m4---m2------m3--|
       |__________________________|
       
    只需要把原来m2负责的部分分出来给m4即可，与其他机器没关系。删除m4也一样，只还回m2即可。
    
问题：
（1）机器少的时候，一上来，环可能做不到均分。
（2）即使开始均衡，都是1/3，加个m4也不均衡。

解决办法：虚拟结点技术！！！

每个机器分1000个str
m1(a1, ..., a1000),   m2(b1,...,b1000),    m3(c1,...,c1000)

就是指定一些代表点增大基数，这3000个点就基本将环等分了。key来了以后能算出所属虚拟结点，也就直接对应到3台机器。实现均衡！

假如假如m4，也分1000个，m4(d1,...,d4)，环上是4000的等分！在小结点上进行数据迁移即可，删除也一样。




'''