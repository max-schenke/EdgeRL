function [] = generateXilinxNN()

global n_bits bin_pt mux_width neuron_latency
n_bits = 48;
bin_pt = 42;
mux_width = 32;
neuron_latency = 7;

level = "SimulinkModelFilename";
architecture.nb_neurons = 90;
architecture.nb_layers = 9;
architecture.nb_inputs = 18;
architecture.nb_outputs = 8;

try
    delete_block(strcat(level, '/Layer'))
    delete_block(strcat(level, '/MainClock'))
catch
end
addMainClock(level, architecture)
addLayer(level, architecture)


        
function [] = addMainClock(level, architecture)
    global n_bits neuron_latency
    add_block('simulink/Ports & Subsystems/Subsystem', strcat(level, '/MainClock'))
    level = strcat(level, '/MainClock');
    delete_line(level, 'In1/1', 'Out1/1')
    delete_block(strcat(level, '/In1'))
    delete_block(strcat(level, '/Out1'))
    
    add_block('xbsControl_r4/Counter', strcat(level, '/clock'), 'arith_type', 'Unsigned', 'cnt_type', 'Free Running', 'n_bits', string(max([ceil(log2(architecture.nb_inputs)), ceil(log2(architecture.nb_outputs)), ceil(log2(architecture.nb_neurons))])), 'bin_pt', '0', 'rst', 'on', 'period', '1e-8', 'en', 'on')
    add_block('xbsBasic_r4/Mux', strcat(level, '/layer_select'), 'latency', '0', 'inputs', '2', 'Precision', 'User Defined', 'arith_type', 'Unsigned', 'n_bits', string(n_bits), 'bin_pt', '0')
    add_block('xbsBasic_r4/Constant', strcat(level, '/input_len'), 'const', string(architecture.nb_inputs), 'arith_type', 'Unsigned', 'n_bits', string(n_bits), 'bin_pt', '0')    
    add_block('xbsBasic_r4/Constant', strcat(level, '/hidden_len'), 'const', string(architecture.nb_neurons), 'arith_type', 'Unsigned', 'n_bits', string(n_bits), 'bin_pt', '0')
    add_block('xbsBasic_r4/Constant', strcat(level, '/nb_layers'), 'const', string(architecture.nb_layers), 'arith_type', 'Unsigned', 'n_bits', string(n_bits), 'bin_pt', '0')
    add_block('xbsBasic_r4/Relational', strcat(level, '/is_hidden_layer_check'), 'latency', '0', 'mode', 'a>b')
    add_block('xbsBasic_r4/Constant', strcat(level, '/input_layer_nb'), 'const', '0', 'arith_type', 'Unsigned', 'n_bits', string(n_bits), 'bin_pt', '0')
    
    add_line(level, 'input_len/1', 'layer_select/2')
    add_line(level, 'hidden_len/1', 'layer_select/3')
    
    add_block('xbsBasic_r4/Relational', strcat(level, '/clock_reset'), 'latency', '0', 'mode', 'a=b', 'op_type', 'Ufix')
    add_line(level, 'layer_select/1', 'clock_reset/1')
    add_line(level, 'clock/1', 'clock_reset/2')
    
    add_block('xbsMath_r4/Accumulator', strcat(level, '/layer_counter'), 'latency', '1', 'hasbypass', 'off', 'rst', 'on', 'n_bits', string(n_bits))
    add_block('xbsBasic_r4/Convert', strcat(level, '/convert_for_rst'), 'latency', '0', 'arith_type', 'Bool')
    add_line(level, 'clock_reset/1', 'convert_for_rst/1')
    add_line(level, 'convert_for_rst/1', 'clock/1')
    add_line(level, 'clock_reset/1', 'layer_counter/1')
    
    add_line(level, 'layer_counter/1', 'is_hidden_layer_check/1', 'autorouting', 'smart')
    add_line(level, 'input_layer_nb/1', 'is_hidden_layer_check/2', 'autorouting', 'smart')
    add_line(level, 'is_hidden_layer_check/1', 'layer_select/1', 'autorouting', 'smart')
    
    add_block('xbsBasic_r4/Relational', strcat(level, '/last_layer'), 'latency', '0', 'mode', 'a=b', 'op_type', 'Bool')
    add_block('xbsBasic_r4/Logical', strcat(level, '/layer_counter_reset'), 'latency', '0', 'logical_function', 'AND', 'precision', 'User Defined', 'arith_type', 'Bool', 'n_bits', '1')
    add_line(level, 'last_layer/1', 'layer_counter_reset/1')
    add_line(level, 'layer_counter_reset/1', 'layer_counter/2')
    add_line(level, 'layer_counter/1', 'last_layer/1')
    add_line(level, 'nb_layers/1', 'last_layer/2')
    add_line(level, 'convert_for_rst/1', 'layer_counter_reset/2')
    
    add_block('simulink/Ports & Subsystems/Out1', strcat(level, '/neuron_count'))
    add_block('simulink/Ports & Subsystems/Out1', strcat(level, '/layer_count'))
    add_line(level, 'layer_counter/1', 'layer_count/1')
    add_line(level, 'clock/1', 'neuron_count/1')
    
    add_block('xbsMath_r4/CMult', strcat(level, '/nb_neurons'), ...
        'latency', '0', ...
        'const', string(architecture.nb_neurons+1), ...
        'const_n_bits', string(n_bits), 'const_bin_pt', '0', ...
        'precision', 'User Defined', 'n_bits', string(n_bits), 'bin_pt', '0', 'arith_type', 'Unsigned')
    add_block('xbsMath_r4/AddSub', strcat(level, '/add'), 'latency', '0', ...
        'n_bits', string(n_bits), 'bin_pt', '0', 'arith_type', 'Unsigned', ...
        'Precision', 'User Defined')
    add_line(level, 'nb_neurons/1', 'add/1')
    
    add_block('simulink/Ports & Subsystems/Out1', strcat(level, '/weight_address'))
    add_line(level, 'layer_counter/1', 'nb_neurons/1')
    add_line(level, 'clock/1', 'add/2')
    add_line(level, 'add/1', 'weight_address/1')
    
    add_block('simulink/Ports & Subsystems/Out1', strcat(level, '/is_hidden_layer'))
    add_line(level, 'is_hidden_layer_check/1', 'is_hidden_layer/1')
    
    add_block('simulink/Ports & Subsystems/Out1', strcat(level, '/layer_reset'))
    add_block('simulink/Ports & Subsystems/Out1', strcat(level, '/network_reset'))
    add_line(level, 'convert_for_rst/1', 'layer_reset/1')
    add_line(level, 'layer_counter_reset/1', 'network_reset/1')
    
    % a delay circuit that stops the main clock for the neuron_latency steps that a
    % neuron needs to evaluate the input
    add_block('xbsControl_r4/Counter', strcat(level, '/delay_clock'), 'arith_type', 'Unsigned', 'cnt_type', 'Free Running', 'n_bits', '3', 'bin_pt', '0', 'rst', 'on', 'period', '1e-8', 'en', 'on')
    add_block('xbsBasic_r4/Relational', strcat(level, '/check_delay_finished'), 'latency', '0', 'mode', 'a>=b')
    add_block('xbsBasic_r4/Constant', strcat(level, '/neuron_delay_len'), 'const', string(neuron_latency-1), 'arith_type', 'Unsigned', 'n_bits', string(n_bits), 'bin_pt', '0')
    add_block('xbsBasic_r4/Inverter', strcat(level, '/inverter'), 'latency', '0')
    add_block('xbsBasic_r4/Logical', strcat(level, '/or'), 'latency', '0', 'logical_function', 'OR', 'precision', 'User Defined', 'arith_type', 'Bool', 'n_bits', '1')
    add_line(level, 'delay_clock/1', 'check_delay_finished/1')
    add_line(level, 'neuron_delay_len/1', 'check_delay_finished/2')
    add_line(level, 'check_delay_finished/1', 'clock/2')
    add_line(level, 'check_delay_finished/1', 'inverter/1')
    add_line(level, 'convert_for_rst/1', 'or/1')
    add_line(level, 'convert_for_rst/1', 'delay_clock/1')
    add_line(level, 'inverter/1', 'or/2')
    add_line(level, 'or/1', 'delay_clock/2')
        
       
    
function [] = addLayer(level, architecture)
    global n_bits bin_pt neuron_latency
    add_block('simulink/Ports & Subsystems/Subsystem', strcat(level, '/Layer'))
    level = strcat(level, '/Layer');
    delete_line(level, 'In1/1', 'Out1/1')
    delete_block(strcat(level, '/In1'))
    delete_block(strcat(level, '/Out1'))
    
    % add mulitplexer to select output of each neuron individually
    addMultiMux(level, architecture.nb_neurons+1)
    for i=1:architecture.nb_neurons
        addNeuron(level, i-1, architecture)
        add_line(level, strcat('Neuron', string(i-1), '/1'), strcat('MultiMux/', string(i)), 'autorouting', 'smart')
    end
    add_block('xbsBasic_r4/Constant', strcat(level, '/bias'), 'const', '1', 'n_bits', string(n_bits), 'bin_pt', string(bin_pt))
    add_line(level, 'bias/1', strcat('MultiMux/', string(architecture.nb_neurons+1)), 'autorouting', 'smart')
    
    % add necessary controlling interface
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/input'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/weight'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/updated_layer'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/updated_neuron'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/neuron_clock'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/layer_clock'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/weight_address'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/is_hidden_layer'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/layer_done'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/network_done'))
    
    % check whether the currently evaluated layer contains updatable 
    % neurons
    add_block('xbsBasic_r4/Relational', strcat(level, '/check_layer_update'), 'latency', '0', 'mode', 'a=b')
    add_block('xbsBasic_r4/Delay', strcat(level, '/delay6'), 'latency', '3')
    add_line(level, 'layer_clock/1', 'delay6/1', 'autorouting', 'smart')
    add_line(level, 'delay6/1', 'check_layer_update/2', 'autorouting', 'smart')
    
    add_block('xbsBasic_r4/Convert', strcat(level, '/cast_bool2int'), 'latency', '3', 'n_bits', '1', 'bin_pt', '0', 'arith_type', 'Unsigned')
    add_block('xbsMath_r4/Accumulator', strcat(level, '/switch'), 'latency', '1', 'n_bits', '1', 'rst', 'off')
    add_block('xbsBasic_r4/Convert', strcat(level, '/cast_int2bool'), 'latency', '0', 'n_bits', '1', 'bin_pt', '0', 'arith_type', 'Unsigned')
    add_block('xbsBasic_r4/Mux', strcat(level, '/cancel_layer_mux'), 'latency', '0', 'inputs', '2', 'Precision', 'User Defined', 'arith_type', "Signed", 'n_bits', string(n_bits), 'bin_pt', '0')
    add_block('xbsBasic_r4/Constant', strcat(level, '/cancel_layer_nb'), 'const', string(architecture.nb_layers+1), ...
        'arith_type', 'Unsigned', 'n_bits', string(n_bits), 'bin_pt', string(bin_pt))
    add_line(level, 'network_done/1', 'cast_bool2int/1', 'autorouting', 'smart')
    add_line(level, 'cast_bool2int/1', 'switch/1', 'autorouting', 'smart')
    add_line(level, 'switch/1', 'cast_int2bool/1', 'autorouting', 'smart')
    add_line(level, 'cast_int2bool/1', 'cancel_layer_mux/1', 'autorouting', 'smart')
    add_line(level, 'updated_layer/1', 'cancel_layer_mux/2', 'autorouting', 'smart')
    add_line(level, 'cancel_layer_nb/1', 'cancel_layer_mux/3', 'autorouting', 'smart')
    add_line(level, 'cancel_layer_mux/1', 'check_layer_update/1', 'autorouting', 'smart')
    
    % cast the weight address from the clock to a format the RAM blocks
    % will accept
    add_block('xbsBasic_r4/Convert', strcat(level, '/cast_weight_addr'), 'latency', '0', 'n_bits', string(ceil(log2((architecture.nb_neurons + 1) * (architecture.nb_layers + 1)))), 'bin_pt', '0', 'arith_type', 'Unsigned')
    add_block('xbsBasic_r4/Delay', strcat(level, '/delay5'), 'latency', '3')
    add_line(level, 'weight_address/1', 'delay5/1', 'autorouting', 'smart')
    add_line(level, 'delay5/1', 'cast_weight_addr/1', 'autorouting', 'smart')
    
    % cast the neuron clock signal to a format the neuron-output
    % multiplexer will accept
    add_block('xbsBasic_r4/Convert', strcat(level, '/cast_output_sel'), 'latency', '0', 'n_bits', string(ceil(log2(architecture.nb_neurons+1))), 'bin_pt', '0', 'arith_type', 'Unsigned')
    add_block('xbsBasic_r4/Delay', strcat(level, '/delay7'), 'latency', '3')
    add_line(level, 'neuron_clock/1', 'delay7/1', 'autorouting', 'smart')
    add_line(level, 'delay7/1', 'cast_output_sel/1', 'autorouting', 'smart')
    
    add_line(level, 'cast_output_sel/1', strcat('MultiMux/', string(architecture.nb_neurons + 2)), 'autorouting', 'smart')
    
    % the layer_done signal needs to arrive at the neuron output register
    % at the same time as the corresponding calculation results
    add_block('xbsBasic_r4/Delay', strcat(level, '/delay3'), 'latency', string(neuron_latency + 1))
    add_line(level, 'layer_done/1', 'delay3/1', 'autorouting', 'smart')
    
    % the network_done signal needs to arrive at the network output register
    % at the same time as the corresponding calculation results
    add_block('xbsBasic_r4/Delay', strcat(level, '/delay3_'), 'latency', string(neuron_latency + 1))
    add_line(level, 'network_done/1', 'delay3_/1', 'autorouting', 'smart')
    
    % the first layer takes an external input while all consecutive layers
    % take the output of the preceding layers as input
    % to accomodate for that we add a multiplexer and layer checkup to
    % handle the first layer differently 
    add_block('xbsBasic_r4/Delay', strcat(level, '/delay8'), 'latency', '1')
    add_block('xbsBasic_r4/Mux', strcat(level, '/input_select'), 'latency', '0', 'inputs', '2', 'Precision', 'User Defined', 'arith_type', "Signed", 'n_bits', string(n_bits), 'bin_pt', string(bin_pt))
    add_block('xbsBasic_r4/Delay', strcat(level, '/delay4'), 'latency', '3')
    add_line(level, 'is_hidden_layer/1', 'delay4/1', 'autorouting', 'smart')
    add_line(level, 'delay4/1', 'input_select/1', 'autorouting', 'smart')
    add_line(level, 'input/1', 'input_select/2', 'autorouting', 'smart')
    add_line(level, 'MultiMux/1', 'input_select/3', 'autorouting', 'smart')
    add_line(level, 'input_select/1', 'delay8/1', 'autorouting', 'smart')
    
    %%%%%% NEW %%%%%%%
    % add layer-long delay to the neuron update enable, otherwise neurons
    % could get wrong weights
    add_block('xbsBasic_r4/Delay', strcat(level, '/layer_delay'), 'latency', string(architecture.nb_neurons+4))
    add_block('xbsBasic_r4/Relational', strcat(level, '/relational'), ...
        'latency', '0', 'mode', 'a=b')
    add_line(level, 'updated_neuron/1', 'layer_delay/1', 'autorouting', 'smart')
    add_line(level, 'updated_neuron/1', 'relational/1', 'autorouting', 'smart')
    add_line(level, 'layer_delay/1', 'relational/2', 'autorouting', 'smart')
    
    
    % connect each neuron with the corresponding control signals
    for i=1:architecture.nb_neurons
        add_line(level, 'delay8/1', strcat('Neuron', string(i-1), '/1'), 'autorouting', 'smart')
        add_line(level, 'cast_weight_addr/1', strcat('Neuron', string(i-1), '/2'), 'autorouting', 'smart')
        add_line(level, 'weight/1', strcat('Neuron', string(i-1), '/3'), 'autorouting', 'smart')
        add_line(level, 'check_layer_update/1', strcat('Neuron', string(i-1), '/4'), 'autorouting', 'smart')
        add_line(level, 'updated_neuron/1', strcat('Neuron', string(i-1), '/5'), 'autorouting', 'smart')
        add_line(level, 'relational/1', strcat('Neuron', string(i-1), '/6'), 'autorouting', 'smart')
        add_line(level, 'delay3/1', strcat('Neuron', string(i-1), '/7'), 'autorouting', 'smart')
        add_line(level, 'delay3_/1', strcat('Neuron', string(i-1), '/8'), 'autorouting', 'smart')
    end
    
    % connect the output neurons to the output registers
    add_block('xbsBasic_r4/Delay', strcat(level, '/delay1'), 'latency', '2')
    add_line(level, strcat('delay3_/1'), strcat('delay1/1'), 'autorouting', 'smart')
    for i=1:architecture.nb_outputs
        add_block('simulink/Ports & Subsystems/Out1', strcat(level, '/output', string(i-1)))
        add_block('xbsMemory_r4/Register', strcat(level, '/output_reg', string(i-1)), 'init', '0', 'en', 'on')
        add_line(level, strcat('Neuron', string(i-1), '/1'), strcat('output_reg', string(i-1), '/1'), 'autorouting', 'smart')
        add_line(level, strcat('output_reg', string(i-1), '/1'), strcat('output', string(i-1), '/1'), 'autorouting', 'smart')
        add_line(level, strcat('delay1/1'), strcat('output_reg', string(i-1), '/2'), 'autorouting', 'smart')
    end
    
    
  
    
function [] = addNeuron(level, number, architecture)
    global n_bits bin_pt neuron_latency
    add_block('simulink/Ports & Subsystems/Subsystem', strcat(level, '/Neuron', string(number)))
    level = strcat(level, '/Neuron', string(number));
    
    % give each neuron a number to identify it in case of applicable 
    % updates, also check if it is the correct layer
    add_block('xbsBasic_r4/Constant', strcat(level, '/neuronNumber'), 'const', string(number), ...
        'arith_type', 'Unsigned', 'n_bits', string(n_bits), 'bin_pt', '0')
    add_block('xbsBasic_r4/Relational', strcat(level, '/relational'), ...
        'latency', '0', 'mode', 'a=b')
    add_block('xbsBasic_r4/Logical', strcat(level, '/and'), 'latency', '0', ...
    'logical_function', 'AND', 'precision', 'User Defined', 'arith_type', ...
    'Bool', 'n_bits', '1', 'inputs', '3')
    %add_block('xbsBasic_r4/Delay', strcat(level, '/delay1'), 'latency', string(architecture.nb_neurons+4))
    
    % MAC unit (Multiply-Accumulate), computes scalar product of two
    % vectors
    add_block('xbsMath_r4/Accumulator', strcat(level, '/integrator'), ...
        'latency', '1', 'hasbypass', 'on', 'rst', 'on', 'n_bits', string(n_bits))
    add_block('xbsMath_r4/Mult', strcat(level, '/mult'), ...
        'latency', '3', 'precision', 'User Defined', ...
        'n_bits', string(n_bits), 'bin_pt', string(bin_pt))
    
    % the RAM contains the neuron weights
    add_block('xbsMemory_r4/Single Port RAM', strcat(level, '/ram'), ...
        'latency', '1', 'Depth', string((architecture.nb_neurons + 1) * (architecture.nb_layers + 1)), ...
        'initVector', strcat('zeros(', string((architecture.nb_neurons + 1) * (architecture.nb_layers + 1)), ',1)'), ...
        'optimize', 'Speed')
    
    % neuron output register saves the result of each neuron
    add_block('xbsMemory_r4/Register', strcat(level, '/reg'), ...
        'init', '0', 'en', 'on')
    
    % add an activation function that can be bypassed in the output layer
    addLeakyRelu(level)
    add_block('xbsBasic_r4/Mux', strcat(level, '/activation_mux'), 'latency', '1', 'Precision', 'User Defined', 'arith_type', "Signed", 'n_bits', string(n_bits), 'bin_pt', string(bin_pt))
    
    % signal routing
    delete_line(level, 'In1/1', 'Out1/1')
    delete_block(strcat(level, '/In1'))
    delete_block(strcat(level, '/Out1'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/input'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/weight_address'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/weight'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/layer_update_enable'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/neuron_update_nb'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/neuron_update_enable'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/done_and_reset'))
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/select_activation'))
    add_block('simulink/Ports & Subsystems/Out1', strcat(level, '/output'))
    add_line(level, 'select_activation/1', 'activation_mux/1', 'autorouting', 'smart')
    add_line(level, 'LeakyRelu/1', 'activation_mux/2', 'autorouting', 'smart')
    add_line(level, 'integrator/1', 'activation_mux/3', 'autorouting', 'smart')
    add_line(level, 'activation_mux/1', 'reg/1', 'autorouting', 'smart')
    add_line(level, 'integrator/1', 'LeakyRelu/1', 'autorouting', 'smart')
    add_line(level, 'mult/1', 'integrator/1', 'autorouting', 'smart')
    add_line(level, 'input/1', 'mult/1', 'autorouting', 'smart')
    add_line(level, 'reg/1', 'output/1', 'autorouting', 'smart')
    add_line(level, 'ram/1', 'mult/2', 'autorouting', 'smart')
    
    add_block('xbsBasic_r4/Delay', strcat(level, '/delay2'), 'latency', string(neuron_latency-1))
    add_line(level, 'done_and_reset/1', 'delay2/1', 'autorouting', 'smart')
    add_line(level, 'delay2/1', 'integrator/2', 'autorouting', 'smart')
    
    add_block('xbsBasic_r4/Delay', strcat(level, '/reg_enable_delay'), 'latency', '1')
    add_line(level, 'done_and_reset/1', 'reg_enable_delay/1', 'autorouting', 'smart')
    add_line(level, 'reg_enable_delay/1', 'reg/2', 'autorouting', 'smart')
    
    add_line(level, 'weight_address/1', 'ram/1', 'autorouting', 'smart')
    add_line(level, 'weight/1', 'ram/2', 'autorouting', 'smart')
    add_line(level, 'neuron_update_nb/1', 'relational/1', 'autorouting', 'smart')
    add_line(level, 'neuronNumber/1', 'relational/2', 'autorouting', 'smart')
    add_line(level, 'relational/1', 'and/1', 'autorouting', 'smart')
    add_line(level, 'layer_update_enable/1', 'and/2', 'autorouting', 'smart')
    add_line(level, 'neuron_update_enable/1', 'and/3', 'autorouting', 'smart')
    add_line(level, 'and/1', 'ram/3', 'autorouting', 'smart')
    
    
function [] = addLeakyRelu(level)
    global n_bits bin_pt
    add_block('simulink/Ports & Subsystems/Subsystem', strcat(level, '/LeakyRelu'))
    level = strcat(level, '/LeakyRelu');
    
    % LeakyRelu implements a case distinction y=max(alpha*x, x) for
    % 0<alpha<1
    add_block('xbsBasic_r4/Mux', strcat(level, '/mux'), 'latency', '0', 'Precision', 'User Defined', 'arith_type', "Signed", 'n_bits', string(n_bits), 'bin_pt', string(bin_pt))
    add_block('xbsBasic_r4/Relational', strcat(level, '/relational'), ...
        'latency', '0', ...
        'mode', 'a<b')
    add_block('xbsMath_r4/CMult', strcat(level, '/gain'), ...
        'latency', '0', ...
        'const', 'alpha', ...
        'const_n_bits', string(n_bits), 'const_bin_pt', string(bin_pt), ...
        'precision', 'User Defined', 'n_bits', string(n_bits), 'bin_pt', string(bin_pt))
    
    % signal routing
    delete_line(level, 'In1/1', 'Out1/1')
    add_line(level, 'gain/1', 'mux/3', 'autorouting', 'smart')
    add_line(level, 'relational/1', 'mux/1', 'autorouting', 'smart')
    add_line(level, 'In1/1', 'mux/2', 'autorouting', 'smart')
    add_line(level, 'In1/1', 'gain/1', 'autorouting', 'smart')
    add_line(level, 'gain/1', 'relational/2', 'autorouting', 'smart')
    add_line(level, 'In1/1', 'relational/1', 'autorouting', 'smart')
    add_line(level, 'mux/1', 'Out1/1', 'autorouting', 'smart')
        
        
  function [] = addMultiMux(level, nb_inports)
    global mux_width n_bits bin_pt
    add_block('simulink/Ports & Subsystems/Subsystem', strcat(level, '/MultiMux'))
    level = strcat(level, '/MultiMux');
    delete_line(level, 'In1/1', 'Out1/1')
    delete_block(strcat(level, '/In1'))
    
    % a multimux cascades as many multiplexers as are necessary to distinct
    % nb_inports signals, which is necessary if there is a hardware
    % restriction to the number of output ports of a single multiplexer
    mux_counter = 0;
    inports_left = nb_inports;
    
    % add the corresponding number of inports
    for i=1:nb_inports
        add_block('simulink/Ports & Subsystems/In1', strcat(level, '/in', string(i-1)))
    end
    
    % add the necessary number of input multiplexers and route the inports
    % accordingly
    while inports_left > mux_width
        add_block('xbsBasic_r4/Mux', strcat(level, '/inmux', string(mux_counter)), 'latency', '0', 'inputs', string(mux_width), 'Precision', 'User Defined', 'arith_type', "Signed", 'n_bits', string(n_bits), 'bin_pt', string(bin_pt))
        for i=1:mux_width
            add_line(level, strcat('in', string(i + mux_counter * mux_width - 1), '/1'), strcat('inmux', string(mux_counter), '/', string(i + 1)), 'autorouting', 'smart')
        end
        mux_counter = mux_counter + 1;
        inports_left = inports_left - mux_width + 1;
    end
    % add one output multiplexer (unconditional, one will at least need this one)
    add_block('xbsBasic_r4/Mux', strcat(level, '/inmux', string(mux_counter)), 'latency', '0', 'inputs', string(inports_left), 'Precision', 'User Defined','arith_type', "Signed", 'n_bits', string(n_bits), 'bin_pt', string(bin_pt))
    
    
    if mux_counter == 0
        % if there is only one multiplexer necessary hard restrictions are
        % not yet met
        left_lines = nb_inports;
    else
        % if we needed multiple multiplexers then some routing was already
        % done
        left_lines = mod(nb_inports, mux_width);
    end
    for i=1:left_lines
        add_line(level, strcat('in', string(i + mux_counter * mux_width - 1), '/1'), strcat('inmux', string(mux_counter), '/', string(mux_counter+1+i)), 'autorouting', 'smart')
    end
    
    % route input multiplexers to output multiplexer (if necessary)
    for i=1:mux_counter
        add_line(level, strcat('inmux', string(i-1), '/1'), strcat('inmux', string(mux_counter), '/', string(i+1)), 'autorouting', 'smart')
    end
    add_line(level, strcat('inmux', string(mux_counter), '/1'), 'Out1/1', 'autorouting', 'smart')
    
    % add a selector block that splits the input selector signal and
    % applies a corresponding selector to each individual mux
    addSelector(level, mux_counter+1, nb_inports)
    for i=1:mux_counter+1
        add_line(level, strcat('Selector/', string(i)), strcat('inmux', string(i-1), '/1'), 'autorouting', 'smart')
    end
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/sel'))
    add_line(level, 'sel/1', 'Selector/1', 'autorouting', 'smart')
    
    
function [] = addSelector(level, mux_counter, nb_inports)
    global mux_width n_bits bin_pt
    add_block('simulink/Ports & Subsystems/Subsystem', strcat(level, '/Selector'))
    level = strcat(level, '/Selector');
    delete_line(level, 'In1/1', 'Out1/1')
    delete_block(strcat(level, '/In1'))
    delete_block(strcat(level, '/Out1'))
    % add a selector block that splits the selector signal for a multimux
    
    % add input and one output for each mux
    add_block('simulink/Ports & Subsystems/In1', strcat(level, '/mastersel'))
    for i=1:mux_counter
        add_block('simulink/Ports & Subsystems/Out1', strcat(level, '/sel', string(i-1)))
    end
    
    if mux_counter > 1
        output_mux_inputs = floor(nb_inports / mux_width) + mod(nb_inports, mux_width);
        add_block('xbsBasic_r4/Convert', strcat(level, '/modulo'), 'latency', '0', 'n_bits', string(ceil(log2(mux_width))), 'bin_pt', '0', 'arith_type', 'Unsigned')
        add_line(level, 'mastersel/1', 'modulo/1', 'autorouting', 'smart')
        add_block('xbsControl_r4/Shift', strcat(level, '/divider'), 'shift_bits', string(ceil(log2(mux_width))), 'shift_dir', 'right', 'latency', '0', 'precision', 'User Defined', 'n_bits', string(ceil(log2(output_mux_inputs))), 'bin_pt', '0', 'arith_type', 'Unsigned')
        add_line(level, 'mastersel/1', 'divider/1', 'autorouting', 'smart')
        for i=1:mux_counter-1
            add_line(level, 'modulo/1', strcat('sel', string(i-1), '/1'), 'autorouting', 'smart')
        end
        if mod(nb_inports, mux_width) == 0
            add_line(level, 'divider/1', strcat('sel', string(mux_counter-1), '/1'), 'autorouting', 'smart')  
        else
            add_block('xbsBasic_r4/Mux', strcat(level, '/selector_mux'), 'latency', '0', 'inputs', '2', 'arith_type', "Unsigned", 'n_bits', string(n_bits), 'bin_pt', string(bin_pt))
            add_block('xbsMath_r4/AddSub', strcat(level, '/add'), 'latency', '0', 'n_bits', string(ceil(log2(output_mux_inputs))), 'bin_pt', '0', 'arith_type', 'Unsigned', 'Precision', 'User Defined')
            add_line(level, 'modulo/1', 'add/1', 'autorouting', 'smart')
            add_line(level, 'add/1', 'selector_mux/3', 'autorouting', 'smart')
            add_line(level, 'divider/1', 'add/2', 'autorouting', 'smart')
            add_line(level, 'divider/1', 'selector_mux/2', 'autorouting', 'smart')
            add_block('xbsBasic_r4/Constant', strcat(level, '/full_muxes'), 'const', string(floor(nb_inports / mux_width)), 'arith_type', 'Unsigned', 'n_bits', string(n_bits), 'bin_pt', '0')
            add_block('xbsBasic_r4/Relational', strcat(level, '/from_overfull_inports'), 'latency', '0', 'mode', 'a=b')
            add_line(level, 'full_muxes/1', 'from_overfull_inports/1', 'autorouting', 'smart')
            add_line(level, 'divider/1', 'from_overfull_inports/2', 'autorouting', 'smart')
            add_line(level, 'from_overfull_inports/1', 'selector_mux/1', 'autorouting', 'smart')
            add_line(level, 'selector_mux/1', strcat('sel', string(mux_counter-1), '/1'), 'autorouting', 'smart')
        end
    else
        % one mux is the easiest case, we only need one selector signal
        add_block('xbsBasic_r4/Convert', strcat(level, '/cast'), 'latency', '0', 'n_bits', string(ceil(log2(nb_inports))), 'bin_pt', '0', 'arith_type', 'Unsigned')
        add_line(level, 'mastersel/1', 'cast/1', 'autorouting', 'smart')
        add_line(level, 'cast/1', 'sel0/1', 'autorouting', 'smart')
    end
