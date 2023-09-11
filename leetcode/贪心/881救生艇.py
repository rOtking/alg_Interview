class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        people = sorted(people)

        i, j = 0, len(people) - 1
        boat = 0

        while(i < j):
            if people[i] + people[j] > limit:
                j -= 1
            else:
                i += 1
                j -= 1
            boat += 1

        if i == j:
            boat += 1

        return boat


# todo ok! 自己做出来了！原来这就是贪心啊！ hahah