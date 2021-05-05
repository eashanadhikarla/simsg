from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

ea = event_accumulator.EventAccumulator('events.out.tfevents.1620138295.magic01')
ea.Reload()
ea.Tags()

len_train = len(ea.Scalars('Train_bbox_pred'))
train_bbox = []
val_bbox = []
for i in range (len_train):
    train_bbox.append(ea.Scalars('Train_bbox_pred')[i][2])
    val_bbox.append(ea.Scalars('Val_bbox_pred')[i][2])
    
# G_loss = []
# D_loss = []
# len_gen = len(ea.Scalars("G_g_gan_obj_loss"))

# for j in range(len_gen):
#     G_loss.append(ea.Scalars('G_g_gan_img_loss')[j][2])
#     D_loss.append(ea.Scalars('D_img_d_img_gan_loss')[j][2])

# total_loss = []
# real_loss = []
# fake_loss = []
# len_gen = len(ea.Scalars("G_total_loss"))

# for j in range(len_gen):
#     total_loss.append(ea.Scalars('G_total_loss')[j][2])
#     real_loss.append(ea.Scalars('D_obj_d_ac_loss_real')[j][2])
#     fake_loss.append(ea.Scalars('D_obj_d_ac_loss_fake')[j][2])

    
x = np.linspace(0, len_train, len_train)


plt.plot(x, train_bbox, ".-", c='b')
#plt.plot(x, val_bbox, ".-" , c='g')
plt.title('CRN BBox Loss')
plt.legend(["Train BBOX Predction", "Val BBOX Predction"])
plt.xlabel('Times/Four Epochs')
plt.ylabel('LOSS')
plt.grid(True)
plt.savefig('crn_Bbox_b.png')
# plt.show()

# y = np.linspace(0, len_gen, len_gen)


# plt.plot(y, G_loss, ".-", c='r')
# plt.plot(y, D_loss, ".-" , c='y')
# plt.title('Geneartion Gan Image Loss vs Discriminator Image Loss')
# plt.legend(["Geneartion Gan Image Loss", "Discriminator Image Loss"])
# plt.xlabel('Epochs')
# plt.ylabel('LOSS')
# plt.grid(True)
# plt.savefig('crn_gen_loss.png')



# x = np.linspace(0, len_gen, len_gen)


# plt.plot(x, total_loss, ".-", c='b')
# plt.plot(x,real_loss, ".-" , c='g')
# plt.plot(x,fake_loss,".-", c ='r')
# plt.title('CRN GAN Loss')
# plt.legend(["Generator Loss", "Discirminator Loss Real", "Discirminator Loss Fake"])
# plt.xlabel('EPOCHS')
# plt.ylabel('LOSS')
# plt.grid(True)
# plt.savefig('crn_GAN.png')
# plt.show()