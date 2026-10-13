import control as ctrl
import matplotlib.pyplot as plt

wn,zeta = 2.0, 0.5

# Transfer Function 构造传递函数
sys = ctrl.tf([wn**2], [1, 2*zeta*wn, wn**2]) 

time,response = ctrl.step_response(sys)
plt.plot(time, response)
plt.title('Step Response of Second-Order System')
plt.xlabel('Time (s)')
plt.ylabel('Response')
# grid用于设置图表的网格线，可以提高图表的可读性
plt.grid(True)
plt.show()