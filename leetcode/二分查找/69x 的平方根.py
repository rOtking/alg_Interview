class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0
        l, r = 0, x
        while(l <= r):
            mid = l + int((r - l) / 2)
            if mid * mid <= x and (mid + 1) * (mid + 1) > x:
                return mid
            elif mid * mid < x:
                l = mid + 1   # todo l与r一定要变化，mid可能是0，不能让l与r不变，会死循环！ 哈哈
            elif mid * mid > x:
                r = mid - 1

if __name__ == '__main__':
    s = Solution()
    res = s.mySqrt(16)
    print(res)


# ok
# ez