class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 可以拉平，一个生序。m行n列
        m, n = len(matrix), len(matrix[0])

        l, r = 0, m * n - 1
        while(l <= r):
            mid = l + int((r - l) / 2)
            x = int(mid / n)
            y = int(mid % n)
            if matrix[x][y] == target:
                return True
            elif matrix[x][y] < target:
                l = mid + 1
            elif matrix[x][y] > target:
                r = mid - 1
            else:
                pass

        return False


# ok
# ez!哈哈