import matplotlib.pyplot as plt 
import numpy as np  

a =[] 
rewards = np.load("320-trajs-rewards.npy")
# plt.hist(a, bins =  list(range(0,101,10))) 
# plt.title("histogram") 
# plt.savefig("histogram_ten.png", dpi=300)
# plt.show()
print(rewards.shape)
for i in range(len(rewards)-30):
    a.append(np.mean(rewards[i:i+30]))

plt.hist(np.array(a), [0,0.03,0.1,0.2])
plt.title("histogram") 
plt.savefig("histogram_context_style.png")
plt.show()
# set up logging