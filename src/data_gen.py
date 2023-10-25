import numpy as np
from qiskit import QuantumCircuit

from time import time
import os

from copy import deepcopy

from quantum_gates import MrAndersonSimulator

class EstimatorZ:

    def __init__(self, simulator:MrAndersonSimulator, qubits_layout:list, device_param:dict):
        
        self.sim = simulator
        self.layout = qubits_layout
        self.dev_params = device_param
        self.n = len(qubits_layout)

    def run(self, circ:QuantumCircuit, shots:int, f:float, idxs:list):
        
        psi0 = np.zeros(2**self.n)
        psi0[0] = 1.0
        self.probs, self.vars = self.sim.run(circ, self.layout, psi0, shots, self.dev_params, self.n, f)

        return {"".join(["Z" if i in idx else "I" for i in self.layout]):(self._expval_Z(idx), self._var_Z(shots)) for idx in idxs}
    
    def _var_Z(self, shots):
        return np.sum(self.vars)/shots
    
    def _expval_Z(self, idx:list):
        
        exp_val = 0.0

        for k, p in enumerate(self.probs):
            bin = format(k, "b").zfill(self.n)
            signs = [(1-int(bin[i])*2) for i in idx]
            exp_val += np.prod(signs)*p

        return exp_val

def ansatz_gen(theta, random_basis_change, num_qubits, num_layers):
    circ = QuantumCircuit(num_qubits, num_qubits)

    for l in range(num_layers):
        for i in range(num_qubits):
            circ.ry(theta[i,l], i)

        for i in range(num_qubits):
            circ.cx(i, (i+1)%num_qubits)
    
    for i in range(num_qubits):
        
        circ.rx(random_basis_change[0,i], i)
        circ.rz(random_basis_change[1,i], i)
        circ.rx(random_basis_change[2,i], i)

    return circ

def general_setup(qubits_layout, device_name, num_layers):
    
    from qiskit import transpile
    from quantum_gates.circuits import EfficientCircuit
    from quantum_gates.utilities import DeviceParameters, setup_backend

    # SIMULATOR SETUP
    simulator = MrAndersonSimulator(CircuitClass=EfficientCircuit)

    # BACKEND SETUP
    backend = setup_backend(
        Token="27cbe88a0e3bb035e3135d4a12a98dc9f58e35b766ad3bfa3da5971f60145e0e56f38d205a38ac22015cd3204a1f8ee7857435cf55aba17c5dda350679f10b0f",
        group="open", project="main", hub="ibm-q", device_name=device_name
    )

    device_param = DeviceParameters(qubits_layout)
    device_param.load_from_backend(backend)
    
    dict_device_param = device_param.__dict__()
    #dict_device_param['T1']=[0 for q in qubits_layout]
    #dict_device_param['T2']=[0 for q in qubits_layout]

    #ESTIMATOR SETUP
    estimator = EstimatorZ(simulator, qubits_layout,  dict_device_param)

    # ANSATZ SETUP
    def sample_circuit():

        n = len(qubits_layout)  #number of qubits
        L = num_layers          #number of layers

        theta = 2*np.pi*np.random.rand(n,L)
        random_basis_change = 2*np.pi*np.random.rand(3,n)
    
        ansatz = transpile(ansatz_gen(theta, random_basis_change, n, L), 
                        backend=backend, 
                        scheduling_method="asap", 
                        initial_layout=qubits_layout, 
                        seed_transpiler=69)
        return ansatz

    return estimator, sample_circuit

def save_data(spec, data, it0, itf):
    cwd = os.getcwd()

    import pickle
    with open(os.path.join(cwd, f"zne_data_{it0}-{itf}_"+"_".join(k+str(val) for k, val in spec.items())), "wb") as dump_file:
        pickle.dump(data, dump_file)

class TimeAndDump():

    def __init__(self, spec, data, it0):
        self.spec = spec
        self.data = data
        self.start_iteration = it0

    def __enter__(self):
        self.t0 = time()
    
    def __exit__(self, *e):

        dt = time()-self.t0
        print(f"Elapsed time:{dt}s...", end=" ")
        
        data_volume = len(self.data)
        save_data(self.spec, self.data, self.start_iteration, self.start_iteration+data_volume)
        print(f"Succesfully dumped {data_volume} random curves to {os.getcwd()}.")

def main():

    qubits_layout = [0,1]
    device_name = "ibm_lagos"

    shots = 2

    num_layers = 2

    # GENERAL SETUP
    estimator, sample_circuit = general_setup(qubits_layout, device_name, num_layers)

    #NOISE SCALING SETUP
    start = -3 # i.e. start at 10^-3
    end = 2 # i.e. end at 10^2
    num_fs = 10 # number of fs to be simulated
    
    fs = np.logspace(start, end, num_fs)

    #DATA AMOUNT
    n_curves = 1
    idxs = [[i] for i in qubits_layout] + [[i,j] for i in qubits_layout for j in qubits_layout[(i+1):]]

    #ACTUAL ITERATION
    print(f"Data generation started:\nNumber of curves: {n_curves}")

    data = []
    spec = {"q":len(qubits_layout), "l":num_layers}
    with TimeAndDump(spec, data, 0):

        for _ in range(n_curves):

            circ = sample_circuit()
            data.append([[f, estimator.run(deepcopy(circ), shots, f, idxs)] for f in fs])
            print(data[-1])
            
if __name__ == "__main__":
    main()