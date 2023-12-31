''''''

# 1.二维堆
'''
愿问题：100亿url的大文件，每个url占64B，找出所有重复的URL。

可以布隆，边添加边查询，重复的记下来。

可用hash分流多个小文件。


补充问题：搜索某公司一天的用户搜索词是百亿级别，设计每天求top100的方法。
--------------------------------------------------

hash成100个小文件，每个小文件用map统计出本小文件的top100，搞个大根堆。

把100个heap的堆顶拿出来，单独做成一个大根堆。

流程：
    （1）每次pop一个，就是第一名；
    （2）看pop的数来自哪个堆，就把那个堆的堆顶去掉，把新堆顶加入总堆做调整。
    （3）总堆pop100次即可。
    
二维堆方便信息汇总，且调整代价位O(logn)，很低。


排名问题：

        二维堆，先解决局部小排名，再汇总为大排名。

'''
# -------------------------------------#

# 2.分段统计 范围统计
'''
原问题：32位无符号，范围0 ~ 2^32-1（40多亿）,有个含有40亿数的文件。所以一定有数没出现过，10G内存找出所有出现两次的数。

（1）万能的hash分流，分小文件，用map统计。

 todo....
（2）位图的升级。

补充：最多10kB内存，找这401亿个数的中位数。从小到大第20亿个数是多少。

范围统计的思想:

1.看10KB能申请多大的无符号整型数组：10KB / 4B = 25K   一个int是4B，那arr长度就是25K = 25 * 1024

2.看2的几次方离25K最近，应该是2048=2^11，把数组的长度定为2048

3.数的范围range是2^32个数，把范围等分为   2048份，每份的范围是2^32 / 2^11 = 2^21的大小

即 范围1：0 ~ 2^11-1，范围2：2^11 ~ 2^12-1，....直到2^32 - 1

遍历40亿数，当前数在哪个范围，arr[i] += 1    得到每个范围的数量。

如arr = [10亿, 8亿,5亿,...]   

可见前两个18亿<20，前三个23>20，所求就是arr[3]的第2亿个数！

在arr【3】的范围上重复上述过程，直到得到结果。

'''

# -------------------------------------#

# 3.题
'''
10G文件，硬盘上，是无序的数。通过什么手段，只用5G内存，输出另一个文件，让其有序。

其实5K都可以。

堆技巧？

还是按内存，把范围分了。

5K / 4B 的大小假设是x，容量x的arr，桶排序的思想，把数的范围range分成一个个的x，计数。然后输出排序，统计下一份。

不知大左神没什么一定用堆。感觉桶排序就行！不纠结，先继续吧





'''