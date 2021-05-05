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
                 n = 1000,
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
        self.Pe = self._Pe_calculation(self.U, self.L, self.D)

        # calculate betas
        self.betas = self.eigenvalues()

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
        n = 1000, number of terms to use in series solution.
        '''
        print('''
              Default parameters for the demo are:
              solute influent concentration C0 = 100 mg/L,
              flow velocity in column U = 10 cm/h,
              dispersion coeffiecient D = 100 cm2/h,
              length of column L = 30 cm,
              number of terms to use in series solution n = 1000.
              ''')
        self.characteristic_equation()
        self.eigenvalues()
        self.concentration_profile()
        self.effluent_concentration(time_end = 12, interval = 0.1)

    def _Pe_calculation(self, U, L, D):
        return U * L / D

    def _characteristic(self, Pe, beta):
        # Define the characteristic equation function
        return beta * np.cos(beta) / np.sin(beta) - beta ** 2/Pe + Pe/4

    def _characteristic_one_para(self, beta):
        return beta * np.cos(beta) / np.sin(beta) - beta**2/self.Pe + self.Pe/4



    def characteristic_equation(self,                                
                                plot = False,
                                savefig = False,
                                savefig_dpi = 200,
                                figsize = None,
                                dpi = None,
                                chara_color = 'r',
                                chara_width = 1.5,
                                singul_color = 'k',
                                singul_width = 1):
        # Make a list of the first few singularities
        singularities = [np.pi * i for i in range(11)]
        
        if self.demo_plot == True or plot == True:
            # Make a customized plot area
            fig, ax = plt.subplots(figsize=figsize, dpi = dpi)
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
                ax.plot(xs,ys, c = chara_color, lw = chara_width)
                ax.axvline(x=s2, c = singul_color, ls = '--', lw = singul_width)

            # make a formatted manual legend
            ls = [lines.Line2D([-1],[-1], c = chara_color, lw = chara_width),
            lines.Line2D([-1],[-1], c = singul_color, ls = '--', lw = singul_width)]
            labels = ['Characteristic Equation','Singularities']
            leg = plt.legend(loc = 2, facecolor = 'white', framealpha = 1, handles = ls, labels = labels)
            leg.get_frame().set_edgecolor('k')
            plt.show()
            
            if dpi != None:
                savefig_dpi = dpi
            if self.demo_plot_save == True:
                plt.savefig('characteristic_equation', dpi = savefig_dpi)
            elif savefig != False:
                if savefig == True:
                    plt.savefig('characteristic_equation', dpi = savefig_dpi)
                else:
                    plt.savefig(str(savefig), dpi = savefig_dpi)

    def eigenvalues(self,
                     print_betas = False):
        # Make a list of the intervals to look for each value of beta
        intervals = [np.pi * i for i in range(self.n)]
        # Store the eigenvalues in a list
        betas = []
        # iterate through the interval and find the beta value
        for i in range(len(intervals) - 1):
            mi = intervals[i] + 10**-10
            ma = intervals[i+1] - 10**-10

            # Brent's method can find the value of the
            # characteristic equation within a given interval
            betas.append(optimize.brentq(self._characteristic_one_para, mi, ma))

        if print_betas == True:
            print('betas are:\n', betas)

        return betas

    def _eigenfunction(self, Pe, B, x, t):
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
                              figsize = None, 
                              dpi = None,
                              print_conc = False,
                              savefig = False,
                              savefig_dpi = 200):

        # Define an array of x values to estimate the function
        column_positions = self.L * np.array(positions)

        # Store the results at each x,t pair in a list of lists
        Cs= []
        
        # flags for concentrations that higher than C0 or lower than 0
        C_high = 0
        C_low = 0
        
        # Estimate the concentration for each dimensionless time and x
        for t in times:
            tau = self.D * t / self.L ** 2
            Cs.append([])
            for p in column_positions:
                x = p / self.L

                # Get the eigenfunction values for all the eigenvalues
                series = self._eigenfunction(self.Pe, np.array(self.betas), x, tau)

                # Sum the series and convert the result to concentration at the point
                C = self.C0 * (1 - 2 * self.Pe *
                               np.exp(self.Pe/2 * x - self.Pe**2/4 * tau) *
                               series.sum())
                
                if C > self.C0:
                    C = self.C0
                    C_high += 1
                elif C < 0:
                    C = 0
                    C_low +=1
                    
                Cs[-1].append(C)
        
        if C_high > 0:
            print('WARNING: {} concentration(s) higher than C0 for concentration profile'.format(C_high))
        if C_low > 0:
            print('WARNING: {} concentration(s) lower than 0 for concentration profile'.format(C_low))
        
        
        if print_conc == True:
            print(Cs)

        if self.demo_plot == True or plot == True:
            self._concentration_profile_plot(Cs, times, column_positions, 
                                             figsize, dpi, savefig, savefig_dpi)
            
        return Cs
    
    def _concentration_profile_plot(self, 
                                    Cs, 
                                    times, 
                                    column_positions, 
                                    figsize = None, 
                                    dpi = None, 
                                    savefig = False,
                                    savefig_dpi = 200):
        
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
        ax.set_xlabel('Position in column (cm)', size = 12, weight = 'bold')
        ax.set_ylabel('Concentration (mg/L)', size = 12, weight = 'bold')
        ax.set_title('Column Concentration Profiles', size = 14, weight = 'bold')
        plt.ylim(-1, self.C0 + 1)
        
        for t, C in zip(times,Cs):
            ax.plot(column_positions, C, label = 't = {:.1f} h'.format(t))
        leg = ax.legend(bbox_to_anchor = (1.02, 0.5), loc = 6, fontsize = 12)
        leg.get_frame().set_linewidth(0)
        plt.show()
        
        if dpi != None:
                savefig_dpi = dpi
                
        if self.demo_plot_save == True:
            plt.savefig('concentration_profile', dpi = savefig_dpi, bbox_inches='tight')
        elif savefig != False:
            if savefig == True:
                plt.savefig('concentration_profile', dpi = savefig_dpi, bbox_inches='tight')
            else:
                plt.savefig(str(savefig), dpi = savefig_dpi, bbox_inches='tight')        
        
        
        
    def effluent_concentration(self,
                               time_end,
                               interval,
                               time_start = 0,
                               plot = False,
                               figsize = None,
                               dpi = None,
                               print_conc = False,
                               savefig = False,
                               savefig_dpi = 200):
        # Define an array time points to estimate the function
        # time_end and interval are required
        self.interval = interval
        times = np.arange(time_start, time_end, self.interval)

        # Store the results in a list
        Cs = []
        
        # flags for concentrations that higher than C0 or lower than 0
        C_high = 0
        C_low = 0
        
        # Estimate the concentration for each dimensionless time at x = 1
        for t in times:
            tau = self.D * t / self.L**2
            x = 1

            # Get the eigenfunction values for all the eigenvalues
            series = self._eigenfunction(self.Pe, np.array(self.betas), x, tau)

            # Sum the series and convert the result to concentration at the point
            C = self.C0 * (1 - 2 * self.Pe * np.exp(self.Pe/2 * x - self.Pe**2/4 * tau) * series.sum())
            
            if C > self.C0:
                C = self.C0
                C_high += 1
            elif C < 0:
                C = 0
                C_low +=1
                
            Cs.append(C)
        
        if C_high > 0:
            print('WARNING: {} concentration(s) higher than C0 for effluent concentration'.format(C_high))
        if C_low > 0:
            print('WARNING: {} concentration(s) lower than 0 for effluent concentration'.format(C_low))
        
        if print_conc == True:
            print(Cs)

        if self.demo_plot == True or plot == True:
            self._breakthrough_curve_plot(Cs, times, time_end, figsize, 
                                          dpi, savefig, savefig_dpi)
        
        return Cs

    def _breakthrough_curve_plot(self, 
                                 Cs, 
                                 times,
                                 time_end,
                                 figsize = None, 
                                 dpi = None, 
                                 savefig = False,
                                 savefig_dpi = 200):
        
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
        ax.set_xlabel('Time (hr)', size = 12, weight = 'bold')
        ax.set_ylabel('Concentration (mg/L)', size = 12, weight = 'bold')
        ax.set_title('Column Breakthrough Curve', size = 14, weight = 'bold')
        plt.ylim(-1, self.C0 + 1)
        ax.plot(times, Cs, ls = '-', c = 'r', label = 'Breakthrough curve')
        
        # Add a couple of other lines for explanation of behavior
        xs = [0, self.L/self.U, self.L/self.U, time_end]
        ys = [0, 0, self.C0, self.C0]
        ax.plot(xs, ys, ls = '-.', lw = 1, c = 'b', label = 'Plug flow')
        leg = ax.legend()
        plt.show()
        
        if dpi != None:
                savefig_dpi = dpi
                
        if self.demo_plot_save == True:
            plt.savefig('breakthrough_curve', dpi = savefig_dpi)
        elif savefig != False:
            if savefig == True:
                plt.savefig('breakthrough_curve', dpi = savefig_dpi)
            else:
                plt.savefig(str(savefig), dpi = savefig_dpi)        
                
    def get_concentration(self,
                          time):

        time = time

        Cs = []

        
        tau = self.D * time / self.L**2
        x = 1

        # Get the eigenfunction values for all the eigenvalues
        series = self._eigenfunction(self.Pe, np.array(self.betas), x, tau)

            # Sum the series and convert the result to concentration at the point
        C = self.C0 * (1 - 2 * self.Pe * np.exp(self.Pe/2 * x - self.Pe**2/4 * tau) * series.sum())
        
        if C > self.C0:
            C = self.C0
            print('WARNING: concentration higher than C0')
        elif C < 0:
            C = 0
            print('WARNING: concentration lower than 0')
                
        Cs.append(C)

        return Cs[0]

    def _effluent_calculation(self,
                              time_start,
                              time_end,
                              time_size,
                              D):

        times = np.linspace(time_start, time_end, time_size)

        self.Pe = self._Pe_calculation(self.U, self.L, D)
        betas = self.eigenvalues()

        # Store the results in a list
        Cs = []
        
        # flags for concentrations that higher than C0 or lower than 0
        C_high = 0
        C_low = 0
        
        # Estimate the concentration for each dimensionless time at x = 1
        for t in times:
            tau = D * t / self.L**2
            x = 1

            # Get the eigenfunction values for all the eigenvalues
            series = self._eigenfunction(self.Pe, np.array(betas), x, tau)

            # Sum the series and convert the result to concentration at the point
            C = self.C0 * (1 - 2 * self.Pe * np.exp(self.Pe/2 * x - self.Pe**2/4 * tau) * series.sum())
            
            if C > self.C0:
                C = self.C0
            elif C < 0:
                C = 0
                
            Cs.append(C)

        return Cs



    def _MSE(self, x, y):
        squared_difference = [(x_i - y_i) ** 2 for x_i, y_i in zip(x, y)]
        mse = sum(squared_difference)
        return mse

    def fit_D(self,
              time,
              conc,
              max_attempts = 20,
              plot = False,
              figsize = None,
              dpi = None,
              savefig = False,
              savefig_dpi = 200):
        
        D = np.linspace(1, 200, 20)

        time_start = time[0]
        time_end = time[-1]
        time_size = len(time)

        default_D_mse = {}
        for d_values in D:
            Cs = self._effluent_calculation(time_start, time_end, time_size, d_values)
            mse = self._MSE(Cs, conc)
            default_D_mse[d_values] = mse

        default_D_mse = dict(sorted(default_D_mse.items(), key = lambda d: d[1]))
        min_mse_D = list(default_D_mse)[0]
        min_mse_2_D = list(default_D_mse)[1]
        D_mse_dict = {min_mse_D: default_D_mse[min_mse_D],
                      min_mse_2_D: default_D_mse[min_mse_2_D]}


        attempt = 0
        while attempt < max_attempts:
            current_D = (list(D_mse_dict)[0] + list(D_mse_dict)[1]) / 2 # Python 3.6 required
            Cs = self._effluent_calculation(time_start, time_end, time_size, current_D)
            mse = self._MSE(Cs, conc)
            D_mse_dict[current_D] = mse
            D_mse_dict = dict(sorted(D_mse_dict.items(), key = lambda d: d[1]))
            D_pop = list(D_mse_dict)[-1]
            D_mse_dict.pop(D_pop)

            attempt += 1

        result_D = list(D_mse_dict)[0]
        result_mse = D_mse_dict[result_D]
        print('D is {D}\nMSE is {mse}'.format(D = result_D, mse = result_mse))
        
        if plot == True:
            self._fit_plot(time, conc, result_D, figsize, dpi, savefig, savefig_dpi)
            
            
        
    def _fit_plot(self,
                  time,
                  conc,
                  final_D,
                  figsize = None, 
                  dpi = None, 
                  savefig = False,
                  savefig_dpi = 200):
        
        Cs = self._effluent_calculation(time[0], time[-1], len(time), final_D)
        
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
        ax.set_xlabel('Time (hr)', size = 12, weight = 'bold')
        ax.set_ylabel('Concentration (mg/L)', size = 12, weight = 'bold')
        ax.set_title('Raw data and fitted curve', size = 14, weight = 'bold')
        
        if Cs[-1] >= conc[-1]:
            y_lim = Cs[-1]
        else:
            y_lim = conc[-1]
        plt.ylim(-1, y_lim + 1)
        
        ax.scatter(time, conc, label = 'Raw data')
        ax.plot(time, Cs, ls = '-', c = 'r', label = 'Breakthrough curve')
        leg = ax.legend()
        plt.show()
        
        if dpi != None:
                savefig_dpi = dpi
                
        if savefig != False:
            if savefig == True:
                plt.savefig('raw_data_and_breakthrough_curve', dpi = savefig_dpi)
            else:
                plt.savefig(str(savefig), dpi = savefig_dpi)  

















if __name__ == '__main__':
    # c = ColumnTracer(demo = True, demo_plot=True, demo_plot_save=False)
    # c = ColumnTracer()
    # c.concentration_profile(print_conc=True)
    # c.effluent_concentration(12, 0.1, plot = True)
    # eff = c.get_concentration(time=2, interval=0.1)
    # eff = c.get_concentration(time=2)
    # print(eff)










    # # time = [ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ,
    # #     1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2. ,  2.1,
    # #     2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3. ,  3.1,  3.2,
    # #     3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4. ,  4.1,  4.2,  4.3,
    # #     4.4,  4.5,  4.6,  4.7,  4.8,  4.9,  5. ,  5.1,  5.2,  5.3,  5.4,
    # #     5.5,  5.6,  5.7,  5.8,  5.9,  6. ,  6.1,  6.2,  6.3,  6.4,  6.5,
    # #     6.6,  6.7,  6.8,  6.9,  7. ,  7.1,  7.2,  7.3,  7.4,  7.5,  7.6,
    # #     7.7,  7.8,  7.9,  8. ,  8.1,  8.2,  8.3,  8.4,  8.5,  8.6,  8.7,
    # #     8.8,  8.9,  9. ,  9.1,  9.2,  9.3,  9.4,  9.5,  9.6,  9.7,  9.8,
    # #     9.9, 10. , 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9,
    # #     11. , 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9]
    # # conc = [0, 2.0872192862952943e-12, 3.1287579083105754e-06, 0.0010020258925158565, 0.019322359212570195, 0.11836208413558147, 0.4043632723391255, 0.9839664335757781, 1.9304988243840637, 3.2743464067235495, 5.008822532605883, 7.101332765726786, 9.504194775878483, 12.163068855780846, 15.022763303236609, 18.030854958983454, 21.139703900330808, 24.307374798996396, 27.49785777834186, 30.680867813689318, 33.831411255228815, 36.92924195182944, 39.95828347629913, 42.906063031658306, 45.76318237789182, 48.52283820778807, 51.18039640252658, 53.73301983123506, 56.17934668044553, 58.51921494458423, 60.75342817276259, 62.8835575214161, 64.91177539533, 66.84071633747325, 68.67336126786977, 70.41294162352307, 72.06286038644274, 73.62662738979137, 75.10780665614291, 76.50997384492484, 77.83668216937679, 79.09143538945568, 80.27766669948006, 81.39872251161005, 82.45785029213144, 83.4581897403678, 84.40276671299272, 85.29448939234607, 86.13614627852532, 86.93040565367201, 87.67981622486035, 88.38680870091893, 89.05369809975141, 89.68268661743711, 90.27586691959179, 90.8352257399996, 91.36264769211404, 91.85991921628207, 92.32873259999033, 92.77069002050591, 93.18730756936074, 93.5800192265204, 93.95018075905514, 94.29907352491485, 94.62790816719327, 94.93782818820954, 95.22991339597164, 95.5051832182337, 95.76459988150984, 96.0090714541462, 96.23945475394578, 96.45655812194741, 96.66114406482646, 96.8539317690552, 97.03559949046512, 97.20678682322448, 97.3680968525031, 97.52009819526583, 97.66332693373012, 97.79828844605882, 97.92545913884673, 98.04528808590801, 98.15819857779192, 98.26458958634996, 98.36483714855669, 98.45929567365262, 98.54829917753426, 98.63216244816775, 98.71118214565013, 98.78563784038846, 98.85579299271377, 98.92189587709498, 98.98418045396869, 99.04286719205521, 99.09816384388907, 99.15026617715549, 99.19935866429154, 99.24561513268314, 99.28919937766712, 99.33026574042975, 99.368959652782, 99.40541815068367, 99.43977035828794, 99.47213794417998, 99.50263555139136, 99.53137120268455, 99.55844668251865, 99.58395789702841, 99.60799521327446, 99.63064377895152, 99.65198382367451, 99.6720909428995, 99.6910363654761, 99.70888720577153, 99.72570670125343, 99.74155443636712, 99.75648655349629, 99.77055595175042, 99.78381247427993, 99.79630308477965]
    # import pandas as pd
    
    # c = ColumnTracer()  
    # data = pd.read_csv('data.csv')
    # time = data['time'].values.tolist()
    # conc = data['concentration'].values.tolist()
    
    
    # c.fit_D(time, conc, plot=True, dpi = 200)