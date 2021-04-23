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
                series = self._eigenfunction(self.Pe, np.array(self.betas), x, tau)

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
        self.interval = interval
        times = np.arange(time_start, time_end, self.interval)

        # Store the results in a list
        Cs = []

        # Estimate the concentration for each dimensionless time at x = 1
        for t in times:
            tau = self.D * t / self.L**2
            x = 1

            # Get the eigenfunction values for all the eigenvalues
            series = self._eigenfunction(self.Pe, np.array(self.betas), x, tau)

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

        # Estimate the concentration for each dimensionless time at x = 1
        for t in times:
            tau = D * t / self.L**2
            x = 1

            # Get the eigenfunction values for all the eigenvalues
            series = self._eigenfunction(self.Pe, np.array(betas), x, tau)

            # Sum the series and convert the result to concentration at the point
            C = self.C0 * (1 - 2 * self.Pe * np.exp(self.Pe/2 * x - self.Pe**2/4 * tau) * series.sum())
            Cs.append(C)

        return Cs



    def _MSE(self, x, y):
        squared_difference = [(x_i - y_i) ** 2 for x_i, y_i in zip(x, y)]
        mse = sum(squared_difference)
        return mse

    def fit_D(self,
              time,
              conc,
              max_attempts = 20):
        # D = [1, 50, 100, 150, 200]
        D = np.linspace(1, 200, 20)
        # D = [30, 60, 90]

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
        print(default_D_mse)

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














# TODO: make jupyter notebook with examples
# TODO: test files



if __name__ == '__main__':
    # c = ColumnTracer(demo = True, demo_plot=True, demo_plot_save=True)
    # c = ColumnTracer()
    # eff = c.get_concentration(time=2, interval=0.1)
    # eff = c.get_concentration(time=2)
    # print(eff)











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
    conc = [0, 2.0872192862952943e-12, 3.1287579083105754e-06, 0.0010020258925158565, 0.019322359212570195, 0.11836208413558147, 0.4043632723391255, 0.9839664335757781, 1.9304988243840637, 3.2743464067235495, 5.008822532605883, 7.101332765726786, 9.504194775878483, 12.163068855780846, 15.022763303236609, 18.030854958983454, 21.139703900330808, 24.307374798996396, 27.49785777834186, 30.680867813689318, 33.831411255228815, 36.92924195182944, 39.95828347629913, 42.906063031658306, 45.76318237789182, 48.52283820778807, 51.18039640252658, 53.73301983123506, 56.17934668044553, 58.51921494458423, 60.75342817276259, 62.8835575214161, 64.91177539533, 66.84071633747325, 68.67336126786977, 70.41294162352307, 72.06286038644274, 73.62662738979137, 75.10780665614291, 76.50997384492484, 77.83668216937679, 79.09143538945568, 80.27766669948006, 81.39872251161005, 82.45785029213144, 83.4581897403678, 84.40276671299272, 85.29448939234607, 86.13614627852532, 86.93040565367201, 87.67981622486035, 88.38680870091893, 89.05369809975141, 89.68268661743711, 90.27586691959179, 90.8352257399996, 91.36264769211404, 91.85991921628207, 92.32873259999033, 92.77069002050591, 93.18730756936074, 93.5800192265204, 93.95018075905514, 94.29907352491485, 94.62790816719327, 94.93782818820954, 95.22991339597164, 95.5051832182337, 95.76459988150984, 96.0090714541462, 96.23945475394578, 96.45655812194741, 96.66114406482646, 96.8539317690552, 97.03559949046512, 97.20678682322448, 97.3680968525031, 97.52009819526583, 97.66332693373012, 97.79828844605882, 97.92545913884673, 98.04528808590801, 98.15819857779192, 98.26458958634996, 98.36483714855669, 98.45929567365262, 98.54829917753426, 98.63216244816775, 98.71118214565013, 98.78563784038846, 98.85579299271377, 98.92189587709498, 98.98418045396869, 99.04286719205521, 99.09816384388907, 99.15026617715549, 99.19935866429154, 99.24561513268314, 99.28919937766712, 99.33026574042975, 99.368959652782, 99.40541815068367, 99.43977035828794, 99.47213794417998, 99.50263555139136, 99.53137120268455, 99.55844668251865, 99.58395789702841, 99.60799521327446, 99.63064377895152, 99.65198382367451, 99.6720909428995, 99.6910363654761, 99.70888720577153, 99.72570670125343, 99.74155443636712, 99.75648655349629, 99.77055595175042, 99.78381247427993, 99.79630308477965]

    c.fit_D(time, conc)
