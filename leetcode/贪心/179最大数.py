import functools
class Solution:
    # todo 错误的版本
    def largestNumber_wrong(self, nums) -> str:

        res = ''
        nums_str = [str(x) for x in nums]
        nums_str = sorted(nums_str, reverse=True)

        for s in nums_str:
            res += s
        return res

# todo 输入：[3,30,34,5,9]，输出："9534303"，预期："9534330"
# todo 30和3排序时30大，有问题。


    # todo 关键在于自定义sorted的比较函数
    def largestNumber(self, nums) -> str:
        def cmp(str1, str2):
            # todo 不用去尝试完成规则，直接得到结果比较不就好了？穷举也就2个结果啊！哈哈
            res1 = int(str1 + str2)
            res2 = int(str2 + str1)
            if res1 > res2:
                return 1
            elif res1 < res2:
                return -1
            else:
                return 0
        res = ''
        nums_str = [str(x) for x in nums]
        nums_str = sorted(nums_str, key=functools.cmp_to_key(cmp), reverse=True)

        for s in nums_str:
            res += s
        # 处理开头是0
        if res[0] == '0' and len(res) > 0:
            return '0'
        return res

if __name__ == '__main__':
    s = Solution()
    res = s.largestNumber([3,30,34,5,9])
    print(res)

# ok
# todo 在看看，有个印象！
