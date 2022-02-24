import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.widgets import CheckButtons


dic_dermatomes = {0:'Backgroud', 10:'Medial Plantar Pie Derecho', 11:'Medial Plantar Pie Izquierdo', 20:'Lateral Plantar Pie Derecho', 21:'Lateral Plantar Pie Izquierdo',
                  30:'Sural Pie Derecho', 31:'Sural Pie Izquierdo', 40:'Tibial Pie Derecho', 41:'Tibial Pie Izquierdo',
                  50:'Saphenous Pie Derecho', 51:'Saphenous Pie Izquierdo', 255:'Edges'}


derm_id = list(dic_dermatomes.keys())
derm_id.sort()
derm_names = [dic_dermatomes[key] for key in derm_id[1:-1]]


def plot_report(img_temps, segmented_temps, mean_temps, dermatomes_temps, dermatomes_masks, times, path = './outputs/report'):
    exit_code = 0
    num_rows = 3
    num_cols = img_temps.shape[0]
    fig_title = 'Report'
    figsize = 15,8,

    #Definiton of grid layout
    fig = plt.figure(figsize=figsize)
    fig.set_size_inches(figsize[0], figsize[1])
    fig.suptitle(fig_title, fontsize=24, fontweight='bold')

    axs = []
    for i in range(num_rows-1):
        axs_row = []
        for j in range(num_cols):
            axs_row.append(plt.subplot2grid((num_rows,num_cols), (i,j), fig=fig))
        axs.append(axs_row)
        axs_temp = plt.subplot2grid((num_rows,num_cols), (num_rows-1,0), colspan=num_cols-1, fig=fig)

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
                axs[i][j].imshow(img_temps[j,:,:]*(dermatomes_masks[j] != 0), cmap=cmap, norm=norm)
                edges = np.argwhere(dermatomes_masks[j] == 255) #Agregar Line1
                axs[i][j].plot(edges[:,1], edges[:,0], '.w', markersize=1) #Agregar Line2
                # axs[i][j].imshow(segmented_temps[j,:,:], cmap=cmap, norm=norm)
                axs[i][j].axis('off')
            else:
                raise ValueError('Error')

    # ind_t = [0,1,5,10,15]
    for j in range(num_cols):
        axs[0][j].set_title("$t"+"_{"+str(times[j])+"}$", family='serif', size=15, weight=900)

    #Plot mean temperatures
    #print(f"mean temps: {mean_temps}")

    for i,couple in enumerate(mean_temps):
        if type(couple) is not list:
            if not np.isnan(couple):
                mean_temps[i] = [couple, couple]
            else:
                mean_temps[i] = [0, 0]
                print("Warning, nan was fount in temp values")
                exit_code = 1

    mean_temps = np.array(mean_temps)
    
    left_temps = mean_temps[:,0]
    right_temps = mean_temps[:,1]


    colors_ = ['lightcoral', 'firebrick', 'darkcyan', 'mediumaquamarine', 'violet', 'lightsteelblue', 'indianred', 'peru', 'slategray', 'darkolivegreen', 'rosybrown', 'seagreen']
    lines = []
    for i in range(0,len(derm_names)):
        l, = axs_temp.plot(times, dermatomes_temps[:,i], label=derm_names[i], color = colors_[i])
        # plt.legend(loc='upper right')
        lines.append(l)


    l1, = axs_temp.plot(times , left_temps , '-o',label = 'Pie Derecho', color = colors_[-2])
    l2, = axs_temp.plot(times , right_temps , '-o', label = 'Pie Izquierdo', color = colors_[-1])
    lines.append(l1)
    lines.append(l2)
    # axs_temp.set_ylim([np.min(segmented_temps[segmented_temps != 0]), np.max(segmented_temps)])
    # axs_temp.set_yticks(np.linspace(np.min(segmented_temps[segmented_temps != 0]), np.max(segmented_temps),5))
    axs_temp.set_title("Temperatura media de pies")
    axs_temp.set_xlabel("Tiempo (min)")
    axs_temp.set_ylabel("Temperatura (°C)")
    axs_temp.grid(visible= True)
    # axs_temp.legend()
    # fig.tight_layout()
    fig.tight_layout(rect=[0, 0, 0.9, 1])

    rax = plt.axes([0.76, 0.03, 0.2, 0.33])
    labels = [str(line.get_label()) for line in lines]
    visibility = [line.get_visible() for line in lines]
    check = CheckButtons(rax, labels, visibility)

    [rec.set_facecolor(colors_[i]) for i, rec in enumerate(check.rectangles)]
    def func(label):
        index = labels.index(label)
        lines[index].set_visible(not lines[index].get_visible())
        plt.draw()

    check.on_clicked(func)


    #Plot clorbar
    cax = fig.add_axes([axs[-1][-1].get_position().x1 + 0.05,axs[-1][-1].get_position().y0,0.02,axs[0][-1].get_position().y1-axs[-1][-1].get_position().y0])
    #Mappeable objects for connectivities colorbar1
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(segmented_temps)
    cbar = fig.colorbar(sm, cax=cax, ticks=np.linspace(np.min(segmented_temps[segmented_temps != 0]), np.max(segmented_temps), 5))
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(10)

    #Save image

    fig.show()

    format = 'pdf'
    plt.savefig(path+'.'+format,format=format, bbox_inches='tight')
    return exit_code
