import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

#Input Parameters
def plot_report(img_temps, segmented_temps, mean_temps, times, path = './outputs/report'):
    num_rows = 3
    num_cols = img_temps.shape[0]
    fig_title = 'Report'
    figsize = 12,8,

    #Definiton of grid layout
    fig = plt.figure(figsize=figsize)
    fig.suptitle(fig_title, fontsize=24, fontweight='bold')

    axs = []
    for i in range(num_rows-1):
        axs_row = []
        for j in range(num_cols):
            axs_row.append(plt.subplot2grid((num_rows,num_cols), (i,j), fig=fig))
        axs.append(axs_row)
        axs_temp = plt.subplot2grid((num_rows,num_cols), (num_rows-1,0), colspan=num_cols, fig=fig)

    #Plot original input image, segmented image and mean temperatures
    cmap = 'gnuplot'
    norm  = colors.Normalize(vmin=np.min(segmented_temps[segmented_temps != 0]), vmax=np.max(segmented_temps))

    #Plot original input image, segmented image
    for i in range(num_rows-1):
        for j in range(num_cols):
            if i == 0:
                ##FALTA MOSTRAR IMAGEN ORIGINAL
                #FALTA LOS YLABELS DE LAS CFILAS 1 Y 2
                axs[i][j].imshow(img_temps[j,:,:], cmap=cmap, norm=norm)
                axs[i][j].axis('off')
                
            elif i == 1:
                axs[i][j].imshow(segmented_temps[j,:,:], cmap=cmap, norm=norm)
                axs[i][j].axis('off')
            else:
                raise ValueError('Error')

    ind_t = [0,1,5,10,15]
    for j in range(num_cols):
        axs[0][j].set_title("$t"+"_{"+str(times[j])+"}$", family='serif', size=15, weight=900)

    #Plot mean temperatures
    
    mean_temps = np.array(mean_temps)
    
    left_temps = mean_temps[:,0]
    right_temps = mean_temps[:,1]
    
    axs_temp.plot(times , left_temps , '-o', color='salmon',label = 'Pie Izquierdo')
    axs_temp.plot(times , right_temps , '-o', color='blue', label = 'Pie Derecho')
    # axs_temp.set_ylim([np.min(segmented_temps[segmented_temps != 0]), np.max(segmented_temps)])
    # axs_temp.set_yticks(np.linspace(np.min(segmented_temps[segmented_temps != 0]), np.max(segmented_temps),5))
    axs_temp.set_title("Temperatura media de pies")
    axs_temp.set_xlabel("Tiempo (min)")
    axs_temp.set_ylabel("Temperatura (°C)")
    axs_temp.grid()
    axs_temp.legend()
    # fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.tight_layout()

    #Plot clorbar
    cax = fig.add_axes([axs[-1][-1].get_position().x1 + 0.05,axs[-1][-1].get_position().y0,0.02,axs[0][-1].get_position().y1-axs[-1][-1].get_position().y0])
    #Mappeable objects for connectivities colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(segmented_temps)
    cbar = fig.colorbar(sm, cax=cax, ticks=np.linspace(np.min(segmented_temps[segmented_temps != 0]), np.max(segmented_temps), 5))
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(10)

    #Save image
    format = 'pdf'
    plt.savefig(path+'.'+format,format=format, bbox_inches='tight')
    plt.show()