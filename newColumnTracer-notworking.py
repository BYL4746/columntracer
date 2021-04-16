# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt, lines
from scipy import optimize


class ColumnTracer():
    def __init__(self, 
                 C0 = 100, 
                 U = 10,
                 D = 100,
                 L = 30,
                 n = 100,
                 demo = False, 
                 demo_plot = False, 
                 demo_plot_save = False):
        # default parameters
        # solute influent concentration C0 in mg/L
        self.C0 = C0
        # flow velocity in column U in cm/h
        self.U = U
        # dispersion coeffiecient D in cm2/h
        self.D = D
        # length of column L in cm
        self.L = L
        # number of terms to use in series solution
        self.n = n
        
        # calculate Peclet number
        # self.Pe = self._Pe_calculation(self.U, self.L, self.D)
        
        # calculate betas
        # self.eigen_values()
        
        self.demo_choice = demo
        self.demo_plot = demo_plot
        self.demo_plot_save = demo_plot_save
        
        if self.demo_choice == True:
            self._demo_run()
       
    def _demo_run(self):
        '''
        C0 = 100 mg/L, solute influent concentration
        U = 10 cm/h, flow velocity in column
        D = 100 cm2/h, dispersion coeffiecient
        L = 30 cm, length of column
        n = 100, number of terms to use in series solution.
        '''    
        print('''
              Default parameters for the demo are:
              solute influent concentration C0 = 100 mg/L, 
              flow velocity in column U = 10 cm/h, 
              dispersion coeffiecient D = 100 cm2/h, 
              length of column L = 30 cm, 
              number of terms to use in series solution n = 100. 
              ''')
        self.characteristic_equation()
        self.eigen_values()
        self.concentration_profile()
        self.effluent_concentration(time_end = 12, interval = 0.1)
    
    def _Pe_calculation(self, U, L, D):
        return U * L / D
    
    def _characteristic(self, Pe, beta):
        # Define the characteristic equation function
        return beta * np.cos(beta) / np.sin(beta) - beta ** 2/Pe + Pe/4
    
    # def _characteristic_one_para(self, beta):
    #     return beta * np.cos(beta) / np.sin(beta) - beta**2/self.Pe + self.Pe/4
    
    def _characteristic_one_para(self, beta, D):
        return beta * np.cos(beta) / np.sin(beta) - beta**2/self._Pe_calculation(self.U, self.L, D) + self._Pe_calculation(self.U, self.L, D)/4
    
    
    def characteristic_equation(self, 
                                plot = False,
                                savefig = False,
                                savefig_dpi = 200):        
        # Make a list of the first few singularities
        singularities = [np.pi * i for i in range(11)]
        
        if self.demo_plot == True or plot == True:
            # Make a customized plot area
            fig, ax = plt.subplots()
            ax.set_ylim(-10,10)
            ax.set_xlim(0, np.pi * 10)
            ax.axhline(y=0, c = 'k', lw = 1)
            ax.set_xlabel(r'$\beta$', weight = 'bold', fontsize = 14)
            ax.set_ylabel(r'$F(Pe, \beta)=\beta$ $\cot$ $\beta - \frac{\beta}{Pe} + \frac{Pe}{4}$', weight = 'bold', fontsize = 14)
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.set_xticks(singularities)
            ax.set_xticklabels(['{}$\pi$'.format(i) for i in range(len(singularities))])
            ax.set_title('Characteristic Equation for Eigenvalues', weight = 'bold', fontsize = 14)
        
            # Go through each interval (n * pi through (n+1) * pi) to plot
            # the function and singularities
            for i in range(len(singularities)-1):
                s1 = singularities[i]
                s2 = singularities[i+1]
                xs = np.arange(s1 + np.pi/100, s2, np.pi/100)
                ys = self._characteristic(self.Pe, xs)
                ax.plot(xs,ys, c = 'r', lw = 1.5)
                ax.axvline(x=s2, c = 'k', ls = '--', lw = 1)
            
            # add an annotation to point out the first few betas
            arrowprops={'arrowstyle':'-|>','connectionstyle':'arc3,rad=-.4','fc':'k'}
            ax.annotate(r'$\beta_{1}$', xy = [1.54,0], xytext = [1,3], fontsize = 12, arrowprops = arrowprops)
            ax.annotate(r'$\beta_{2}$', xy = [3.88,0], xytext = [4.3,3], fontsize = 12, arrowprops = arrowprops)
            ax.annotate(r'$\beta_{3}$', xy = [6.72,0], xytext = [7.5,3], fontsize = 12, arrowprops = arrowprops)
            
            # make a formatted manual legend
            ls = [lines.Line2D([-1],[-1], c = 'r', lw = 1.5), 
            lines.Line2D([-1],[-1], c = 'k', ls = '--', lw = 1)]
            labels = ['Characteristic Equation','Singularities']
            leg = plt.legend(loc = 2, facecolor = 'white', framealpha = 1, handles = ls, labels = labels)
            leg.get_frame().set_edgecolor('k')
            
            if self.demo_plot_save == True:
                plt.savefig('characteristic_equation', dpi = savefig_dpi)            
            elif savefig != False:
                if savefig == True:
                    plt.savefig('characteristic_equation', dpi = savefig_dpi)
                else:
                    plt.savefig(str(savefig), dpi = savefig_dpi)
                
    # TODO: no _ between eigen and values
    def eigen_values(self,
                     print_betas = False):
        # Make a list of the intervals to look for each value of beta
        intervals = [np.pi * i for i in range(self.n)]
        # Store the eigenvalues in a list
        self.betas = []
        # iterate through the interval and find the beta value
        for i in range(len(intervals) - 1):
            mi = intervals[i] + 10**-10
            ma = intervals[i+1] - 10**-10
            
            # Brent's method can find the value of the 
            # characteristic equation within a given interval
            self.betas.append(optimize.brentq(self._characteristic_one_para, mi, ma))
        
        if print_betas == True:
            print('betas are:\n', self.betas)
        
        return self.betas
    
        # TODO: no _
    def _eigen_function(self, Pe, B, x, t):
        # Define a function to use to compute the value of the "ith" term
        # in the series of eigenfunctions that are summed in the solution
        return (B * (B * np.cos(B * x) + Pe/2 * np.sin(B * x)) / 
                (B**2 + Pe**2/4 + Pe) / (B**2 + Pe**2/4) * np.exp(-1 * B**2 * t))
    
    def concentration_profile(self, 
                              times = [0.00001, 0.1, 0.5, 1, 2, 4, 10],
                              positions = [0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
                                           0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 
                                           0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1], 
                              plot = False,
                              print_conc = False,
                              savefig = False,
                              savefig_dpi = 200):
        
        # Define an array of x values to estimate the function
        column_positions = self.L * np.array(positions)
        
        # Store the results at each x,t pair in a list of lists
        Cs= []
        
        # Estimate the concentration for each dimensionless time and x
        for t in times:
            tau = self.D * t / self.L ** 2
            Cs.append([])
            for p in column_positions:
                x = p / self.L
                
                # Get the eigenfunction values for all the eigenvalues
                series = self._eigen_function(self.Pe, np.array(self.betas), x, tau)
                
                # Sum the series and convert the result to concentration at the point
                C = self.C0 * (1 - 2 * self.Pe * 
                               np.exp(self.Pe/2 * x - self.Pe**2/4 * tau) * 
                               series.sum())
                Cs[-1].append(C)
        
        if print_conc == True:
            print(Cs)
        
        if self.demo_plot == True or plot == True:  
            # Plot the results
            fig, ax = plt.subplots()
            ax.set_xlabel('Position in column (cm)', size = 12, weight = 'bold')
            ax.set_ylabel('Concentration (mg/L)', size = 12, weight = 'bold')
            ax.set_title('Column Concentration Profiles', size = 14, weight = 'bold')
            for t, C in zip(times,Cs):
                ax.plot(column_positions, C, label = 't = {:.1f} h'.format(t))            
            leg = ax.legend(bbox_to_anchor = (1.02, 0.5), loc = 6, fontsize = 12) 
            leg.get_frame().set_linewidth(0)
            
            if self.demo_plot_save == True:
                plt.savefig('concentration_profile', dpi = savefig_dpi, bbox_inches='tight')            
            elif savefig != False:
                if savefig == True:
                    plt.savefig('concentration_profile', dpi = savefig_dpi, bbox_inches='tight')
                else:
                    plt.savefig(str(savefig), dpi = savefig_dpi, bbox_inches='tight')
        
        return Cs
                
    def effluent_concentration(self,                           
                               time_end, 
                               interval,
                               time_start = 0,
                               plot = False,
                               print_conc = False,
                               savefig = False,
                               savefig_dpi = 200):
        # Define an array time points to estimate the function
        # time_end and interval are required
        times = np.arange(time_start, time_end, interval) 
        
        # Store the results in a list
        Cs = []
        
        # Estimate the concentration for each dimensionless time at x = 1
        for t in times:
            tau = self.D * t / self.L**2
            x = 1
            
            # Get the eigenfunction values for all the eigenvalues
            series = self._eigen_function(self.Pe, np.array(self.betas), x, tau)
             
            # Sum the series and convert the result to concentration at the point
            C = self.C0 * (1 - 2 * self.Pe * np.exp(self.Pe/2 * x - self.Pe**2/4 * tau) * series.sum())
            Cs.append(C)
        
        if print_conc == True:
            print(Cs)  
        
        if self.demo_plot == True or plot == True:
            # Plot the results
            fig, ax = plt.subplots()
            ax.set_xlabel('Time (hr)', size = 12, weight = 'bold')
            ax.set_ylabel('Concentration (mg/L)', size = 12, weight = 'bold')
            ax.set_title('Column Breakthrough Curve', size = 14, weight = 'bold')
            ax.plot(times, Cs, ls = '-', c = 'r', label = 'Breakthrough curve')
            
            # Add a couple of other lines for explanation of behavior
            xs = [0, self.L/self.U, self.L/self.U, time_end]
            ys = [0, 0, self.C0, self.C0]
            ax.plot(xs, ys, ls = '-.', lw = 1, c = 'b', label = 'Plug flow')
            ax.text(0.5,65,'Effects of Dispersion', fontsize = 12)
            arrowprops = {'arrowstyle':'<|-|>'}
            ax.annotate('', xy = (5.5,61), xytext = (0.5,61), ha = 'right', va = 'center', arrowprops = arrowprops)
            leg = ax.legend()

            if self.demo_plot_save == True:
                plt.savefig('breakthrough_curve', dpi = savefig_dpi)            
            elif savefig != False:
                if savefig == True:
                    plt.savefig('breakthrough_curve', dpi = savefig_dpi)
                else:
                    plt.savefig(str(savefig), dpi = savefig_dpi)
        return Cs
    
    def _C_calculation(self, t, D):
        x = 1
        return self.C0 * (1 - 2 * self._Pe_calculation(self.U, self.L, D) 
                          * np.exp(self._Pe_calculation(self.U, self.L, D)/2 * x - self._Pe_calculation(self.U, self.L, D)**2/4 * D * t / self.L**2) 
                          * self._eigen_function(self._Pe_calculation(self.U, self.L, D), 
                                                  np.array(self.eigen_values()), 
                                                  x, 
                                                  D * t / self.L**2).sum()
                          )
    
    
    def fit_D(self,
              time,
              conc):
        xdata = time
        ydata = conc
        popt, pcov = optimize.curve_fit(self._C_calculation, xdata, ydata)
        print(popt)
        
        
        
        
        
        
        
        
# TODO: make it available on pip
# TODO: new function: input t, get C
if __name__ == '__main__':
    # c = ColumnTracer(demo = True, demo_plot=True, demo_plot_save=True)
    c = ColumnTracer()
    time = [ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ,
        1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2. ,  2.1,
        2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3. ,  3.1,  3.2,
        3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4. ,  4.1,  4.2,  4.3,
        4.4,  4.5,  4.6,  4.7,  4.8,  4.9,  5. ,  5.1,  5.2,  5.3,  5.4,
        5.5,  5.6,  5.7,  5.8,  5.9,  6. ,  6.1,  6.2,  6.3,  6.4,  6.5,
        6.6,  6.7,  6.8,  6.9,  7. ,  7.1,  7.2,  7.3,  7.4,  7.5,  7.6,
        7.7,  7.8,  7.9,  8. ,  8.1,  8.2,  8.3,  8.4,  8.5,  8.6,  8.7,
        8.8,  8.9,  9. ,  9.1,  9.2,  9.3,  9.4,  9.5,  9.6,  9.7,  9.8,
        9.9, 10. , 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9,
       11. , 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9]
    conc = [-0.014037701182689766, 1.036803976006695e-09, 0.0001976591561758667, 0.013695841526328856, 0.12310304180530585, 0.47803934609247367, 1.2076522681544244, 2.373224945265162, 3.973397933893341, 5.966601377319336, 8.29153417694516, 10.8810472493443, 13.670190429236973, 16.60025268650194, 19.620356993578447, 22.68768049780281, 25.766966617055775, 28.829720430890127, 31.853306086607112, 34.820061569234696, 37.716486342639044, 40.532523828123814, 43.26094263624911, 45.89681134937358, 48.43705757024426, 50.8801006260045, 53.22554743212717, 55.473941830725195, 57.6265588083445, 59.68523614760392, 61.6522371624501, 63.53013915739207, 65.32174311955502, 67.03000089906695, 68.65795676704658, 70.20870077372234, 71.68533177523177, 73.0909283691419, 74.42852628722724, 75.70110104967918, 76.91155489642772, 78.06270718604638, 79.15728759703474, 80.19793158535515, 81.18717765029925, 82.12746604170044, 83.02113860819009, 83.87043954110646, 84.67751681386531, 85.44442415378485, 86.17312341393635, 86.86548723771772, 87.52330192948426, 88.14827046150494, 88.74201556139506, 89.3060828355493, 89.8419438934001, 90.35099944492634, 90.83458235003503, 91.2939606034832, 91.73034024310502, 92.14486817242704, 92.53863489143399, 92.91267713140148, 93.26798039143672, 93.60548137574004, 93.9260703316843, 94.23059328965802, 94.51985420627476, 94.79461701305237, 95.05560757303962, 95.30351554813907, 95.53899618006396, 95.7626719879887, 95.97513438602151, 96.17694522365436, 96.36863825233817, 96.55072052129886, 96.7236737056557, 96.887955369836, 97.04400016919938, 97.1922209926968, 97.3330100492961, 97.46673990080755, 97.59376444364385, 97.71441984194858, 97.82902541442725, 97.93788447711607, 98.04128514422712, 98.1395010891132, 98.23279226730364, 98.3214056034732, 98.40557564411954, 98.48552517764149, 98.56146582343041, 98.63359859150982, 98.70211441418503, 98.76719465109322, 98.82901156897813, 98.88772879744727, 98.94350176190957, 98.9964780948317, 99.04679802639511, 99.09459475558316, 99.13999480267611, 99.18311834408368, 99.22407953039833, 99.26298678850914, 99.29994310857384, 99.3350463166069, 99.36838933340412, 99.40006042048793, 99.4301434137233, 99.45871794522236, 99.48585965412403, 99.5116403868064, 99.53612838706137, 99.55938847673444, 99.58148222730776, 99.60246812288028]
    c.fit_D(time, conc)
    
    
   