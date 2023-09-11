# todo 核心思路：扑克牌，手上的是有序的，每新来一张都依次比较，到达合适的位置，插入。
# todo 具体实现上两种：1.新来的有有序区不断交换，直到合适的位置停止；2.找到合适位置，其他位置移动，再把当前数放进去。
# 其实都一样，就用交换的方式就行！   😄


# 时间O(n^2)，稳定

class Solution:
    def insertSort(self, arr):
        for i in range(1, len(arr)):
            for j in range(i, 0, -1):
                if arr[j] < arr[j - 1]:
                    arr[j], arr[j - 1] = arr[j - 1], arr[j]

        return arr




if __name__ == '__main__':
    arr = [9,1,8,2,6]
    sol = Solution()
    res = sol.insertSort(arr)
    print(res)