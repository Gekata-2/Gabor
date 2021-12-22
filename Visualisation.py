import matplotlib.pyplot as plt
import os


def CompareImages(img1,img2,save=False,dir_path=None,names=['',''],title='',save_as=1):
    plt.figure(figsize=(15,9))

    plt.subplot(121)
    plt.imshow(img1,cmap="gray",vmax=255,vmin=0)
    plt.title(names[0])

    plt.subplot(122)
    plt.imshow(img2,cmap='gray',vmax=255,vmin=0)
    plt.title(names[1])
    plt.suptitle(title)
    if save:
        path=os.path.join(dir_path,names[save_as]+'.png')
        print(path)
        plt.savefig(path)
    plt.show()
