from model import Model 


m = Model()
epoch=2
batch_sizes=[32,128,512,1024,2048,4096,8192]
for i in batch_sizes:
    m.execute(epoch,i)
