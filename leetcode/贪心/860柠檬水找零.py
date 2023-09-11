class Solution:
    def lemonadeChange(self, bills) -> bool:
        # 剩余数量
        five = 0
        ten = 0
        twenty = 0
        res = True

        for bill in bills:
            if bill == 5:
                five += 1
            elif bill == 10:
                ten += 1
                if five > 0:
                    five -= 1
                else:
                    res = False
            else:
                twenty += 1
                if ten > 0 and five > 0:
                    ten -= 1
                    five -= 1
                elif five > 2:
                    five -= 3
                else:
                    res = False
        return res



if __name__ == '__main__':
    bills = [5,5,10,20,5,5,5,5,5,5,5,5,5,10,5,5,20,5,20,5]
    sol = Solution()
    res = sol.lemonadeChange(bills)
    print(res)

# ok
# todo 就是按流程走，这也算贪心？   注意20有 10+5 与 5+5+5 两种组合