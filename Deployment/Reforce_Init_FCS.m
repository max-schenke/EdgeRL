clear all
close all
clc

%% Sampling time
Ts = 50e-6; % s

%% Motor and inverter limitations
% (Motor Type: SEW CM3C80S, 6000 rpm)

motor.lim.i = 16; % A
motor.lim.n_me = 800; % 1/min
motor.lim.T = 10.5; % N*m

motor.nominal.i = 13.0; % 15.9; % A %before 10.1 A
motor.nominal.n_me = 6e+3; % 1/min
motor.nominal.omega_me = motor.nominal.n_me * 2 * pi / 60; % 1/s
motor.nominal.T = 10.5; % N*m %changed in Torque_ref_sampling to 30 % of this value

inverter.lim.u_dc = 625;
inverter.lim.time_over_limit = 1e-3; % s

inverter.mapping = [-1 +1 +1 -1 -1 -1 +1 +1;
                    -1 -1 +1 +1 +1 -1 -1 +1;
                    -1 -1 -1 -1 +1 +1 +1 +1;];% switching states order
                
inverter.dead_time = 2.75e-6; % s                
inverter.i_threshold = 1.5; 
inverter.zero_current_slope = 2/3;
                
inverter.DC_link.low = 55; % V
inverter.DC_link.high = 60; % V

inverter.zero_current_slope = 1;

%% Electrical parameters (guess)
motor.param.p =      4; % 1
motor.param.R_s =  208e-3; % Ohm
motor.param.L_d =    1.44e-3; % H
motor.param.L_q =    1.44e-3; % H
motor.param.psi_p = 112e-3; % Vs
motor.param.U_DC = 50; % V
motor.param.tau_d = motor.param.L_d / motor.param.R_s; % s
motor.param.tau_q = motor.param.L_q / motor.param.R_s; % s

abc2alphabeta0 = [      2/3      -1/3       -1/3;
                          0 1/sqrt(3) -1/sqrt(3);
                  sqrt(2)/3 sqrt(2)/3  sqrt(2)/3];
alphabeta02abc = [  1          0 1/sqrt(2);
                 -1/2  sqrt(3)/2 1/sqrt(2);
                 -1/2 -sqrt(3)/2 1/sqrt(2);];
             
%% static Parameter identification
% snapshot from Testbench:
% A_d = [0.9074 0.0458;
%       -0.0948 0.9916];
%    
% B_d = [ 0.0296 0.0031;
%        -0.0039 0.0371];
%    
% E_d = [0.0722; -1.011];

A_d = [0.896142 0.0486026;
      -0.095931 0.9833810];
   
B_d = [ 0.029023 0.003453;
       -0.005443 0.035227];
   
E_d = [1.072431; -3.149501];

w_el = 2200 * 2*pi/60 * motor.param.p;

ATs = logm(A_d);
A = ATs / Ts;

Bd_tilde = [B_d, E_d];

B_tilde = (A_d - eye(2))^(-1) * A * Bd_tilde;
B = B_tilde(:,1:2);
E = B_tilde(:,3);

motor.ident.L_d = 1/B(1,1);
motor.ident.L_q = 1/B(2,2);
Rs1=-motor.ident.L_d*A(1,1);
Rs2=-motor.ident.L_q*A(2,2);
motor.ident.R_s = (Rs1+Rs2)/2;
motor.ident.psi_p = -E(2)*motor.ident.L_q/w_el;
             
%% Calibration

sensor.current.ph1.gain = 25.595;
sensor.current.ph1.offset = -0.0156;

sensor.current.ph2.gain = 25.62;
sensor.current.ph2.offset = -0.0025;

sensor.current.ph3.gain = 25.62;
sensor.current.ph3.offset = -0.0261;

sensor.current.DClink.gain = 25.478;
sensor.current.DClink.offset = +0.0137;

sensor.voltage.DClink.gain = 792.040;
sensor.voltage.DClink.offset = +6.0627;

sensor.torque.gain = -20;      

poles_positiveDir = [328.711 332.314 328.711 330.996]; %degree
poles_negativeDir = [333.896 331.875 335.127 331.699]; %degree
sensor.angle_calibrate.bias = 360-(mean(poles_negativeDir)+mean(poles_positiveDir)) / 2;
             
%% PLL init

PLL.fn  =  50;           % [Hz] closed-loop nominal frequency of PLL
PLL.d   =  1;            % [] closed-loop damping factor of PLL
PLL.Ki_Ta = (2 * pi * PLL.fn)^2;    % [1/s^2] integral gain of PLL controller
PLL.Kp = 2 * PLL.d * 2 * pi * PLL.fn;   % [1/s] proportional gain of PLL controller

%% FOC CC init

BW_c = 10000;
CC.d.Kp = motor.param.L_d * BW_c;
CC.d.Ki = CC.d.Kp * motor.param.R_s / motor.param.L_d;
CC.q.Kp = motor.param.L_q * BW_c;
CC.q.Ki = CC.q.Kp * motor.param.R_s / motor.param.L_q;

%% DQ-DTC init

controller.DQDTC.i_d_plus = 4;
controller.DQDTC.torque_tolerance = 0.1;


% learning config
controller.epsilon = 0.0;
controller.DQDTC.activate_scheduler = 0; % 0=manual, 1=scheduler
controller.DQDTC.initial_lr = 1e-3;
controller.DQDTC.final_lr = 1e-7;
controller.DQDTC.initial_epsilon = 0.3;
controller.DQDTC.final_epsilon = 0.0;
controller.DQDTC.schedule_time = 10 * 60; % seconds



controller.u_dq_init = [0; 0;];
controller.u_abc_init = [0; 0; 0;];
controller.DQDTC.action_dim = 8;
controller.DQDTC.gamma = 0.85;
controller.DQDTC.reward_interval = (1-controller.DQDTC.gamma);
controller.DQDTC.torque_tolerance_norm = controller.DQDTC.torque_tolerance / motor.nominal.T;
controller.DQDTC.i_d_plus_norm = controller.DQDTC.i_d_plus / motor.lim.i;
controller.DQDTC.i_nominal_norm = motor.nominal.i / motor.lim.i;
controller.DQDTC.n_upper = 675;
controller.DQDTC.n_lower = -controller.DQDTC.n_upper;
torque_ref_sample_upper = 6.5;
torque_ref_sample_lower = -torque_ref_sample_upper;

controller.DQDTC.u_DC_max = motor.param.U_DC * 1.5;
controller.DQDTC.u_DC_min = motor.param.U_DC * 0.5;

controller.RLS.n = 5;
controller.RLS.lambda = 0.99999;
controller.RLS.theta_0 = ones(5,1) * 1e-1  ;
controller.RLS.P_0 = eye(controller.RLS.n) * 1e1;
controller.RLS.xi_0 = [0; 0; 0; 0; 1;];

% which voltage boundary is surveilled by the safeguard:
% controller.safeguard.voltage_boundary = motor.param.U_DC^2/3;
controller.safeguard.voltage_scaling = (2/pi)^2;

controller.DQDTC.LP_filter.passband_edge = 10; %Hz
controller.DQDTC.LP_filter.stopband_edge = 100; %Hz

for i=0:9
    eval(append('init_layer_', string(i), ' = init_ANN(90);'));
end

data_loc = "C:\Users\mschenke\Desktop\DQDTC_weights\train_10min_06\weights_checkpoint_0.hdf5";
info = h5info(data_loc);

for i=0:9
    eval(append('init_layer_', string(i), " = h5read(data_loc, '/w", string(i),"');"));
    eval(append('init_layer_', string(i), " = init_layer_", string(i), "';"));
end


%% Preindl MPC

controller.preindl.LUT = load('Reforce_PMSM_LUTs.mat');
controller.preindl.LUT.x_vec = -16:1:1;
controller.preindl.LUT.y_vec = -16:1:16;

fn = fieldnames(controller.preindl.LUT);
fn = fn(contains(fn, 'map'));
for k = 1:numel(fn)
    eval(['controller.preindl.LUT.', fn{k}, '=', 'FILL_LUT2D(controller.preindl.LUT.x_vec,controller.preindl.LUT.y_vec,controller.preindl.LUT.', fn{k}, ');']);
end
controller.preindl.LUT.psi_p = LUT2D(controller.preindl.LUT.x_vec,controller.preindl.LUT.y_vec,controller.preindl.LUT.Psi_d_map,0,0);

controller.preindl.zeta = 0.9;
controller.preindl.w_boundary = 100;
controller.preindl.lambda_current = 5;
controller.preindl.lambda_limit = 10000;

controller.preindl.model.B_d = Ts * [1/motor.param.L_d 0;
                                     0 1/motor.param.L_q];   
                                 
controller.compensate_deadtime =1;  % 0=no, 1=yes                               


%% U/f init

U_nom = 47;
f_nom = 1000 / 60;
uf_factor = U_nom / f_nom;

%% Load control
% configured relation between desired load speed and DAC output value
load.speed2voltage = 1/6500;
load.config = 0; % 0=constant speed, 1=randomly changing speed
load.T_ref_prob = 0.0001;
load.n_me_prob = 5e-6;
load.acceleration_rate = 80; %400; %80;

%% Small Signal Torque Test

T_ref = [1.29098740464085,-1.68848928878342,-2.42280088777460,0.597303630813709,-2.30665996764717,-1.20523374181224,-1.36744566204100,0.590059045013907,-0.614782567460916,2.16765775597535,-1.94086703472853,1.06206585713053,1.16812850456607,-1.33920410960303,-2.41139378707120,2.18171004583667,-2.10275453167525,1.98851543678797,-1.94034031130232,-1.11137577707515];

function W_ = init_ANN(nb_neurons)
    W = (zeros(nb_neurons, nb_neurons));
    b = (zeros(nb_neurons, 1));
    W_ = [W, b];
end
