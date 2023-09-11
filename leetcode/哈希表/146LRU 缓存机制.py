class BListNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None
        self.pre = None

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.cache = {}
        self.dummy_head = BListNode(0, 0)
        self.dummy_tail = BListNode(0, 0)
        self.dummy_head.next = self.dummy_tail
        self.dummy_tail.pre = self.dummy_head
    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._move_to_end(node)
            return node.value
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self.cache[key] = node
            self._move_to_end(node)
        else:
            node = BListNode(key, value)
            if self.size < self.capacity:
                self.size += 1
            else:
                self._del_head()
            self.cache[key] = node
            self._add_to_end(node)

    def _del_head(self):
        to_del_node = self.dummy_head.next
        next_ = to_del_node.next
        self.dummy_head.next = next_
        next_.pre = self.dummy_head

        self.cache.pop(to_del_node.key)
        del to_del_node

    def _move_to_end(self, node):
        # 已存在的node移动到end
        pre = node.pre
        next_ = node.next

        pre.next = next_
        next_.pre = pre

        self._add_to_end(node)

    def _add_to_end(self, node):
        # 添加一个新的弄的到end
        pre = self.dummy_tail.pre
        pre.next = node
        node.next = self.dummy_tail

        node.pre = pre
        self.dummy_tail.pre = node


if __name__ == '__main__':
    lRUCache = LRUCache(2)
    lRUCache.put(1, 1)
    lRUCache.put(2, 2)
    print(lRUCache.get(1))
    lRUCache.put(3, 3)
    print(lRUCache.get(2))
    lRUCache.put(4, 4)
    print(lRUCache.get(1))
    print(lRUCache.get(3))
    print(lRUCache.get(4))







# todo 之前的想法：hash存key与value，BListNode存key与之自带的顺序。但是put是O（1），get是要遍历List的，O（n）很慢。

# todo 正确：hash中key：node，直接能get到node，起索引的作用；BlistNode保存完整信息，不断的调整优先级！

# ok 注意移到最后与添加新的到最后的区别！hash + 双向list的运用！