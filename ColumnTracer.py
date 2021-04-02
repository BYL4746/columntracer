# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt, lines
from scipy import optimize

class Column_tracer():
    def __init__(self):
        self.input_dict = {'c0': self.C0_input, 'u': self.U_input, 
                           'd': self.D_input, 'l': self.L_input, 
                           'n': self.n_input}
        
        # default parameters
        # solute influent concentration C0 in mg/L
        self.C0 = 100
        # flow velocity in column U in cm/h
        self.U = 10
        # dispersion coeffiecient D in cm2/h
        self.D = 100
        # length of column L in cm
        self.L = 30
        # number of terms to use in series solution
        self.n = 100
        
        self.demo_choice = input('Do you want run a demo? (y/n):')
        if self.demo_choice.lower() == 'y' or self.demo_choice.lower() == 'yes':
            self.demo_run()
        elif self.demo_choice.lower() == 'n' or self.demo_choice.lower() == 'no' or self.demo_choice.lower() == '':
            self.parameter_input()
            self.characteristic_equation()
            self.eigen_values()
            self.concentration_profile()
            self.effluent_concentration()
            
            
    def demo_run(self):        
        '''
        C0 = 100 mg/L, solute influent concentration
        U = 10 cm/h, flow velocity in column
        D = 100 cm2/h, dispersion coeffiecient
        L = 30 cm, length of column
        n = 100, number of terms to use in series solution
        '''
        self.characteristic_equation()
        self.eigen_values()
        self.concentration_profile()
        self.effluent_concentration()
        
    def characteristic(self, Pe, beta):
        # Define the characteristic equation function
        return beta * np.cos(beta) / np.sin(beta) - beta ** 2/Pe + Pe/4
    
    def characteristic_one_para(self, beta):
        return beta * np.cos(beta) / np.sin(beta) - beta**2/self.Pe + self.Pe/4
    
    def characteristic_equation(self):
        # Compute the Peclet number
        self.Pe = self.U * self.L / self.D
        
        # Make a list of the first few singularities
        singularities = [np.pi * i for i in range(11)]
        
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
            ys = self.characteristic(self.Pe, xs)
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
        
    
    def eigen_values(self):
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
            self.betas.append(optimize.brentq(self.characteristic_one_para, mi, ma))
    
        
    
    def eigen_function(self, Pe, B, x, t):
        # Define a function to use to compute the value of the "ith" term
        # in the series of eigenfunctions that are summed in the solution
        return (B * (B * np.cos(B * x) + Pe/2 * np.sin(B * x)) /
            (B**2 + Pe**2/4 + Pe) / (B**2 + Pe**2/4) * np.exp(-1 * B**2 * t)
           )
    
    def concentration_profile(self):
        # Define an array of x values to estimate the function
        positions = self.L * np.array([0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
                          0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 
                          0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1]) 
        
        # Define an array time points to estimate the function
        times = [0.00001, 0.1, 0.5, 1, 2, 4, 10]
        
        # Store the results at each x,t pair in a list of lists
        Cs= []
        
        # Estimate the concentration for each dimensionless time and x
        for t in times:
            tau = self.D * t / self.L**2
            Cs.append([])
            for p in positions:
                x = p / self.L
                
                # Get the eigenfunction values for all the eigenvalues
                series = self.eigen_function(self.Pe, np.array(self.betas), x, tau)
                
                # Sum the series and convert the result to concentration at the point
                C = self.C0 * (1 - 2 * self.Pe * 
                               np.exp(self.Pe/2 * x - self.Pe**2/4 * tau) * 
                               series.sum())
                Cs[-1].append(C)
            
            # Plot the results
        fig, ax = plt.subplots()

        ax.set_xlabel('Position in column (cm)', size = 12, weight = 'bold')
        ax.set_ylabel('Concentration (mg/L)', size = 12, weight = 'bold')
        ax.set_title('Column Concentration Profiles', size = 14, weight = 'bold')
        for t,C in zip(times,Cs):
            ax.plot(positions,C, label = 't = {:.1f} h'.format(t))            
        leg = ax.legend(bbox_to_anchor = (1.02, 0.5), loc = 6, fontsize = 12) 
        leg.get_frame().set_linewidth(0)
        
        
        

    def effluent_concentration(self):
        # Define an array time points to estimate the function
        times = np.arange(0, 12, 0.1)
        
        # Store the results in a list
        Cs = []
        
        # Estimate the concentration for each dimensionless time at x = 1
        for t in times:
            tau = self.D * t / self.L**2
            x = 1
            
            # Get the eigenfunction values for all the eigenvalues
            series = self.eigen_function(self.Pe, np.array(self.betas), x, tau)
             
            # Sum the series and convert the result to concentration at the point
            C = self.C0 * (1 - 2 * self.Pe * np.exp(self.Pe/2 * x - self.Pe**2/4 * tau) * series.sum())
            Cs.append(C)
        
        # Plot the results
        fig, ax = plt.subplots()
        ax.set_xlabel('Time (hr)', size = 12, weight = 'bold')
        ax.set_ylabel('Concentration (mg/L)', size = 12, weight = 'bold')
        ax.set_title('Column Breakthrough Curve', size = 14, weight = 'bold')
        ax.plot(times, Cs, ls = '-', c = 'r', label = 'Breakthrough curve')
        
        # Add a couple of other lines for explanation of behavior
        xs = [0, self.L/self.U, self.L/self.U, 12]
        ys = [0, 0, self.C0, self.C0]
        ax.plot(xs, ys, ls = '-.', lw = 1, c = 'b', label = 'Plug flow')
        ax.text(0.5,65,'Effects of Dispersion', fontsize = 12)
        arrowprops = {'arrowstyle':'<|-|>'}
        ax.annotate('', xy = (5.5,61), xytext = (0.5,61), ha = 'right', va = 'center', arrowprops = arrowprops)
        leg = ax.legend()
        plt.show()
        # plt.savefig('breakthrough')
        
        
    def parameter_input(self):
       
        print('''
              Please input:
              solute influent concentration C0 in mg/L
              flow velocity in column U in cm/h
              dispersion coeffiecient D in cm2/h
              length of column L in cm
              number of terms to use in series solution n
              You can type c0, u, d, l, or n to jump to the corresponding parameter
              Type enter will use the current values
              ''')
        
        self.C0_input()        
        self.U_input()
        self.D_input()
        self.L_input()
        self.n_input()

    def C0_input(self):
        while True:
            user_input = input('Solute influent concentration C0 in mg/L (current value is {}): '.format(self.C0))
            if user_input == '':
                print('C0 is set to {} '.format(self.C0))
                break
            elif user_input.lower() in self.input_dict.keys():
                self.input_dict[user_input.lower()]()
            else:
                try:
                    self.C0 = float(user_input)
                except ValueError:
                    print('Please input a valid value')
                else:
                    print('C0 is set to {} '.format(self.C0))
                    break
        
    def U_input(self):
        while True:
            user_input = input('Flow velocity in column U in cm/h (current value is {}): '.format(self.U))
            if user_input == '':
                print('U is set to {} '.format(self.U))
                break
            elif user_input.lower() in self.input_dict.keys():
                self.input_dict[user_input.lower()]()
            else:
                try:
                    self.U = float(user_input)
                except ValueError:
                    print('Please input a valid value')
                else:
                    print('U is set to {} '.format(self.U))
                    break
    
    def D_input(self):
        while True:
            user_input = input('Dispersion coeffiecient D in cm2/h (current value is {}): '.format(self.D))
            if user_input == '':
                print('D is set to {} '.format(self.D))
                break
            elif user_input.lower() in self.input_dict.keys():
                self.input_dict[user_input.lower()]()
            else:
                try:
                    self.D = float(user_input)
                except ValueError:
                    print('Please input a valid value')
                else:
                    print('D is set to {} '.format(self.D))
                    break
    
    def L_input(self):
        while True:
            user_input = input('Length of column L in cm (current value is {}): '.format(self.L))
            if user_input == '':
                print('L is set to {} '.format(self.L))
                break
            elif user_input.lower() in self.input_dict.keys():
                self.input_dict[user_input.lower()]()
            else:
                try:
                    self.L = float(user_input)
                except ValueError:
                    print('Please input a valid value')
                else:
                    print('L is set to {} '.format(self.L))
                    break
    
    def n_input(self):
        while True:
            user_input = input('Number of terms to use in series solution (current value is {}): '.format(self.n))
            if user_input == '':
                print('n is set to {} '.format(self.n))
                break
            elif user_input.lower() in self.input_dict.keys():
                self.input_dict[user_input.lower()]()
            else:
                try:
                    self.n = int(user_input)
                except ValueError:
                    print('Please input a valid value')
                else:
                    print('n is set to {} '.format(self.n))
                    break
        
        



if __name__ == '__main__':
    c = Column_tracer()
    