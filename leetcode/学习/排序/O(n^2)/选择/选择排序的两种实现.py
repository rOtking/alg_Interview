class Solution:
    # 1. todo 基本：每轮记录一个最值，与首位置交换；明显不稳定。如[2,2,1]，第一次换为[1,2,2]时两个2的顺序就变化了！
    # 但他只是单纯的遍历并记录下来，不会一次次的交换！
    # 每轮选个最小值
    def selectSort(self, arr):
        for i in range(len(arr)):
            min_index = i
            for j in range(i, len(arr)):
                if arr[j] < arr[min_index]:
                    min_index = j

            arr[i], arr[min_index] = arr[min_index], arr[i]

        return arr


    # 2. 二元选择：每轮顺便把最大值也确定了，就加速了一半！
    # ok
    # todo 自己写的版本最容易记了，首尾双指针收缩，完成二元选择排序
    def selectSort2(self, arr):
        start, end = 0, len(arr) - 1
        while(start < end):
            if arr[start] > arr[end]:
                arr[start], arr[end] = arr[end], arr[start]
            min_index = start
            max_index = end
            for i in range(start + 1, end):
                if arr[i] < arr[min_index]:
                    min_index = i
                elif arr[i] > arr[max_index]:
                    max_index = i
                else:
                    continue
            arr[start], arr[min_index] = arr[min_index], arr[start]
            arr[end], arr[max_index] = arr[max_index], arr[end]
            start += 1
            end -= 1

        return arr






if __name__ == '__main__':

    arr = [9,1,8,2,7,3]
    s = Solution()
    res = s.selectSort2(arr)
    print(res)



