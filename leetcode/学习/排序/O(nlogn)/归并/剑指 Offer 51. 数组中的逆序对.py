class Solution:
    def reversePairs(self, nums) -> int:
        if len(nums) == 0:
            return 0

        return self.process(nums, 0, len(nums) - 1)

    # l-r范围上的逆序对个数
    def process(self, arr, left, right):

        if left == right:
            return 0
        mid = left + ((right - left) >> 1)
        l_num = self.process(arr, left, mid)
        r_num = self.process(arr, mid + 1, right)
        merge_num = self.merge(arr, left, mid, right)

        return l_num + r_num + merge_num

    def merge(self, arr, left, mid ,right):
        help_arr = []
        p1 = left
        p2 = mid + 1
        num = 0
        while(p1 <= mid and p2 <= right):
            if arr[p1] > arr[p2]:
                help_arr.append(arr[p1])
                num += (right - p2 + 1)
                p1 += 1
            else:
                help_arr.append(arr[p2])
                p2 += 1
        # 右测都比剩下的p1大 没有逆序对
        while(p1 <= mid):
            help_arr.append(arr[p1])
            p1 += 1

        # 剩下的左都比右边大，但之前已经算过了
        while(p2 <= right):
            help_arr.append(arr[p2])
            p2 += 1

        for i in range(left, right + 1):
            arr[i] = help_arr[i - left]
        return num


# ok !
if __name__ == '__main__':
    arr = [1,3,2,3,1]
    sol = Solution()
    res = sol.reversePairs(arr)
    print(res)