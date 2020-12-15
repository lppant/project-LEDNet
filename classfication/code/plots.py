import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import os



def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies,outputpath):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	#print('I am here ')
	#print(train_losses)
	plt.plot(train_losses,label='Training Loss')
	plt.plot(valid_losses,label='Validation Loss')
	plt.xlabel('epoch')
	plt.ylabel('Loss')
	plt.legend(loc='upper right')
	plt.savefig(os.path.join(outputpath,'./loss.png'))
	plt.clf()
	plt.plot(train_accuracies, label='Training Accuracy')
	plt.plot(valid_accuracies, label='Validation Accuracy')
	plt.xlabel('epoch')
	plt.ylabel('Accuracy')
	plt.legend(loc='upper left')
	plt.savefig(os.path.join(outputpath,'./accuracy.png'))
	plt.clf()

	pass


def plot_confusion_matrix(results, class_names,outputpath):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	y_true = []
	y_pred = []
	for tuple in results :
		y_true.append(tuple[0])
		y_pred.append(tuple[1])

	#cnfn_mat = confusion_matrix(y_true,y_pred)
	#print(cnfn_mat)
	np.set_printoptions(precision=2)
	plot_confusion_matrix_util(y_true=y_true,
						  y_pred=y_pred,
						  classes = class_names,
						  title='Normalized Confusion Matrix')
	plt.savefig(os.path.join(outputpath,'./confusion_matrix.png'))

	pass


#[Reference : https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html]
def plot_confusion_matrix_util(y_true, y_pred, classes,
                          title=None,
                          cmap=plt.cm.Blues):

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True',
           xlabel='Predicted')
	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
	# Loop over data dimensions and create text annotations.
	fmt = '.2f'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	return ax