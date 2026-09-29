import random
import unittest
from common import prepare, naive
from common_fused import train
from common_queue import train as queue_train
from queues import HeapQueue, HighLowQueue
from backends_fused import FusedEbpeEndpoints, FusedPrezzaBitmap, FusedPrezzaHalfword
from backend_linked_fused import FusedLinked12

class FusedTest(unittest.TestCase):
    def test_full_traces(self):
        rng=random.Random(2917)
        cases=[['aaa'],['abababab'],['a'*1024],['a'*64+'b'+'a'*129]]
        for _ in range(200):
            words=[''.join(rng.choices('abcde',k=rng.randrange(1,40))) for _ in range(rng.randrange(1,12))]
            cases.append(words+rng.choices(words,k=4))
        for words in cases:
            inp=prepare(words)
            merges,final=naive(inp,100)
            for cls in [FusedEbpeEndpoints,FusedPrezzaBitmap,FusedPrezzaHalfword,FusedLinked12]:
                result=train(inp,cls,100,capture=True)
                self.assertEqual(result['merges'],merges)
                self.assertEqual(result['final'],final)
            for queue in [HeapQueue,HighLowQueue]:
                result=queue_train(inp,FusedEbpeEndpoints,100,capture=True,queue_class=queue)
                self.assertEqual(result['merges'],merges)
                self.assertEqual(result['final'],final)

if __name__=='__main__':unittest.main()
