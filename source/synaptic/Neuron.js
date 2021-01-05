import Connection, { connections } from './Connection'

let neurons = 0

// 压缩函数
// squashing functions
const squash = {
	// eq. 5 & 5'
	/**
	 * @param {number} x 
	 * @param {boolean} derivate 
	 */
	LOGISTIC: function (x, derivate) {
		var fx = 1 / (1 + Math.exp(-x))
		if (!derivate)
			return fx
		return fx * (1 - fx)
	},
	TANH: function (x, derivate) {
		if (derivate)
			return 1 - Math.pow(Math.tanh(x), 2)
		return Math.tanh(x)
	},
	IDENTITY: function (x, derivate) {
		return derivate ? 1 : x
	},
	HLIM: function (x, derivate) {
		return derivate ? 1 : x > 0 ? 1 : 0
	},
	RELU: function (x, derivate) {
		if (derivate)
			return x > 0 ? 1 : 0
		return x > 0 ? x : 0
	}
}

/**
 * 神经元是神经网络的基本单位。它们可以连接在一起，或者用于门连接其他神经元。
 * 神经元基本上可以执行 4 个操作：项目连接、门连接、激活和传播。
 */
export default class Neuron {
	/**
	 * 压缩函数
	 */
	static squash = squash

	/**
	 * 获取神经元的唯一id
	 */
	static uid() {
		return neurons++
	}

	constructor() {
		/**
		 * 神经元的唯一id
		 */
		this.ID = Neuron.uid()
		/**
		 * 所有连接
		 */
		this.connections = {
			/**
			 * @type {{[x:string]:Connection}} 输入连接
			 */
			inputs: {},
			/**
			 * @type {{[x:string]:Connection}} 输出连接
			 */
			projected: {},
			/** 
			 * @type {{[x:string]:Connection}} 门控
			 */
			gated: {}
		}
		this.error = {
			responsibility: 0,
			projected: 0,
			gated: 0
		}
		this.trace = {
			elegibility: {},
			extended: {},
			influences: {}
		}
		this.state = 0
		this.old = 0
		this.activation = 0
		/**
		 * 神经元的自我连接
		 * 权重为0等于没连接
		 * weight = 0 -> not connected
		 */
		this.selfconnection = new Connection(this, this, 0)
		/**
		 * 挤压功能和偏置
		 * 默认情况下，神经元使用逻辑Sigmoid 作为其挤压/激活功能，以及随机偏置。您可以按照以下方式更改这些属性：
		 * 
		 * var A = new Neuron();
		 * A.squash = Neuron.squash.TANH;
		 * A.bias = 1;
		 * 有 5 个内置的挤压函数，但您也可以创建自己的：
		 * 
		 * 神经元. squash. 逻辑
		 * 神经元. squash. tanH
		 * 神经元. squash. 身份
		 * 神经元. squash. hlim
		 * 神经元. squash. relu
		 * 请参阅此处的更多挤压函数。
		 * https://wagenaartje.github.io/neataptic/docs/methods/activation/
		 */
		this.squash = Neuron.squash.LOGISTIC
		this.neighboors = {}
		/**
		 * 偏置
		 */
		this.bias = Math.random() * .2 - .1
	}

	/**
	 * 激活
	 * activate the neuron -
	 * 当神经元激活时，它会从所有输入连接计算其状态，
	 * 并使用其激活功能压缩它，并返回输出（激活）。
	 * 您可以将激活作为参数提供（对输入层中的神经元很有用）。
	 * 它必须介于 0 和 1 之间的浮点数
	 * var A = new Neuron();
	 * var B = new Neuron();
	 * A.project(B);
	 * A.activate(0.5); // 0.5
	 * B.activate(); // 0.3244554645
	 * @param {number} input 
	 */
	activate(input) {
		// 环境激活（来自输入神经元）
		// activation from enviroment (for input neurons)
		if (typeof input != 'undefined') {
			this.activation = input
			this.derivative = 0
			this.bias = 0
			return this.activation
		}

		// old state
		this.old = this.state

		// eq. 15
		this.state = this.selfconnection.gain * this.selfconnection.weight *
			this.state + this.bias

		for (var i in this.connections.inputs) {
			var input = this.connections.inputs[i]
			this.state += input.from.activation * input.weight * input.gain
		}

		// eq. 16
		this.activation = this.squash(this.state)

		// f'(s)
		this.derivative = this.squash(this.state, true)

		// update traces
		var influences = []
		for (var id in this.trace.extended) {
			// extended elegibility trace
			var neuron = this.neighboors[id]

			// if gated neuron's selfconnection is gated by this unit, the influence keeps track of the neuron's old state
			var influence = neuron.selfconnection.gater == this ? neuron.old : 0

			// index runs over all the incoming connections to the gated neuron that are gated by this unit
			for (var incoming in this.trace.influences[neuron.ID]) { // captures the effect that has an input connection to this unit, on a neuron that is gated by this unit
				influence += this.trace.influences[neuron.ID][incoming].weight *
					this.trace.influences[neuron.ID][incoming].from.activation
			}
			influences[neuron.ID] = influence
		}

		for (var i in this.connections.inputs) {
			let input = this.connections.inputs[i]

			// elegibility trace - Eq. 17
			this.trace.elegibility[input.ID] = this.selfconnection.gain * this.selfconnection
				.weight * this.trace.elegibility[input.ID] + input.gain * input.from
					.activation

			for (var id in this.trace.extended) {
				// extended elegibility trace
				var xtrace = this.trace.extended[id]
				var neuron = this.neighboors[id]
				var influence = influences[neuron.ID]

				// eq. 18
				xtrace[input.ID] = neuron.selfconnection.gain * neuron.selfconnection
					.weight * xtrace[input.ID] + this.derivative * this.trace.elegibility[
					input.ID] * influence
			}
		}

		//  update gated connection's gains
		for (var connection in this.connections.gated) {
			this.connections.gated[connection].gain = this.activation
		}

		return this.activation
	}

	/**
	 * 反向传播
	 * back-propagate the error -
	 * 激活后，您可以教神经元什么应该是正确的输出（即训练）。
	 * 这是通过回侵占错误完成的。
	 * 若要使用传播方法，您必须提供学习速率和目标值（浮动在 0 和 1 之间）。
	 * 例如，当神经元 A 激活 1 时，您可以训练神经元 B 以激活 0：
	 * var A = new Neuron();
	 * var B = new Neuron();
	 * A.project(B);
	 * 
	 * var learningRate = .3;
	 * 
	 * for(var i = 0; i < 20000; i++)
	 * {
	 * 		// when A activates 1
	 * 	 	A.activate(1);
	 * 		// train B to activate 0
	 * 		B.activate();
	 * 		B.propagate(learningRate, 0); 
	 * }
	 * 
	 * // test it
	 * A.activate(1);
	 * B.activate(); // 0.006540565760853365
	 * @param {number} rate 学习速率
	 * @param {number} target 目标值
	 */
	propagate(rate, target) {
		// 误差累加器
		// error accumulator
		var error = 0

		// 这个神经元是否在输出层
		// whether or not this neuron is in the output layer
		var isOutput = typeof target != 'undefined'

		// 输出神经元从环境中获得误差
		// output neurons get their error from the enviroment
		if (isOutput)
			// Eq. 10
			this.error.responsibility = this.error.projected = target - this.activation

		// 其余的神经元通过反向传播计算它们的错误责任
		// the rest of the neuron compute their error responsibilities by backpropagation
		else {

			// 此神经元投射的所有连接的错误责任
			// error responsibilities from all the connections projected from this neuron
			for (var id in this.connections.projected) {
				var connection = this.connections.projected[id]
				var neuron = connection.to
				// Eq. 21
				error += neuron.error.responsibility * connection.gain * connection.weight
			}

			// 预计错误责任
			// projected error responsibility
			this.error.projected = this.derivative * error

			error = 0
			// 此神经元选通的所有连接的错误责任
			// error responsibilities from all the connections gated by this neuron
			for (var id in this.trace.extended) {
				var neuron = this.neighboors[id] // gated neuron
				var influence = neuron.selfconnection.gater == this ? neuron.old : 0 // if gated neuron's selfconnection is gated by this neuron

				// index runs over all the connections to the gated neuron that are gated by this neuron
				for (var input in this.trace.influences[id]) { // captures the effect that the input connection of this neuron have, on a neuron which its input/s is/are gated by this neuron
					influence += this.trace.influences[id][input].weight * this.trace.influences[
						neuron.ID][input].from.activation
				}
				// eq. 22
				error += neuron.error.responsibility * influence
			}

			// gated error responsibility
			this.error.gated = this.derivative * error;

			// error responsibility - Eq. 23
			this.error.responsibility = this.error.projected + this.error.gated;
		}

		// learning rate
		rate = rate || .1;

		// adjust all the neuron's incoming connections
		for (var id in this.connections.inputs) {
			var input = this.connections.inputs[id];

			// Eq. 24
			var gradient = this.error.projected * this.trace.elegibility[input.ID];
			for (var id in this.trace.extended) {
				var neuron = this.neighboors[id];
				gradient += neuron.error.responsibility * this.trace.extended[
					neuron.ID][input.ID];
			}
			input.weight += rate * gradient; // adjust weights - aka learn
		}

		// adjust bias
		this.bias += rate * this.error.responsibility;
	}

	/**
	 * 项目
	 * 神经元可以投射到另一个神经元的连接（即将神经元A与神经元B连接）。以下是其完成：
	 * var A = new Neuron();
	 * var B = new Neuron();
	 * A.project(B); // A now projects a connection to B
	 * 神经元也可以自我连接：
	 * A.project(A);
	 * 方法项目返回一个对象，该对象可由另一个神经元进行门控。Connection
	 * @param {Neuron} neuron 另一个神经元
	 * @param {number} weight 权重(0-1)
	 */
	project(neuron, weight) {
		// 神经元自我连接
		// self-connection
		if (neuron == this) {
			this.selfconnection.weight = 1
			return this.selfconnection
		}

		// 检查连接是否已存在
		// check if connection already exists
		var connected = this.connected(neuron)
		if (connected && connected.type == 'projected') {
			// 更新连接
			// update connection
			if (typeof weight != 'undefined')
				connected.connection.weight = weight
			// return existing connection
			return connected.connection
		} else {
			// 创建新连接
			// create a new connection
			var connection = new Connection(this, neuron, weight)
		}

		// 参考所有的连接和痕迹
		// reference all the connections and traces
		this.connections.projected[connection.ID] = connection
		this.neighboors[neuron.ID] = neuron;
		neuron.connections.inputs[connection.ID] = connection
		neuron.trace.elegibility[connection.ID] = 0

		for (var id in neuron.trace.extended) {
			var trace = neuron.trace.extended[id]
			trace[connection.ID] = 0
		}

		return connection
	}

	/**
	 * 门
	 * 神经元可以门控两个神经元之间的连接，或神经元的自我连接。这允许您创建二阶神经网络体系结构。
	 * var A = new Neuron();
	 * var B = new Neuron();
	 * var connection = A.project(B);
	 * var C = new Neuron();
	 * C.gate(connection); // now C gates the connection between A and B
	 * @param {Connection} connection 
	 */
	gate(connection) {
		// add connection to gated list
		this.connections.gated[connection.ID] = connection

		var neuron = connection.to
		if (!(neuron.ID in this.trace.extended)) {
			// extended trace
			this.neighboors[neuron.ID] = neuron;
			var xtrace = this.trace.extended[neuron.ID] = {};
			for (var id in this.connections.inputs) {
				var input = this.connections.inputs[id];
				xtrace[input.ID] = 0;
			}
		}

		// keep track
		if (neuron.ID in this.trace.influences)
			this.trace.influences[neuron.ID].push(connection);
		else
			this.trace.influences[neuron.ID] = [connection];

		// set gater
		connection.gater = this;
	}

	/**
	 * 无论神经元是否自连接，都返回true或false
	 * returns true or false whether the neuron is self-connected or not
	 */
	selfconnected() {
		return this.selfconnection.weight !== 0
	}

	/**
	 * 检查神经元是否连接到另一个神经元
	 * returns true or false whether the neuron is connected to another neuron (parameter)
	 * @param {Neuron} neuron 另一个神经元
	 */
	connected(neuron) {
		/**
		 * @type {{type:string,connection:Connection}}
		 */
		var result = {
			type: null,
			connection: null
		}

		// 检查是否自我连接
		if (this == neuron) {
			if (this.selfconnected()) {
				result.type = 'selfconnection'
				result.connection = this.selfconnection
				return result
			} else
				return false
		}

		for (var type in this.connections) {
			for (var connection in this.connections[type]) {
				var connection = this.connections[type][connection]
				if (connection.to == neuron) {
					result.type = type
					result.connection = connection
					return result
				} else if (connection.from == neuron) {
					result.type = type
					result.connection = connection
					return result
				}
			}
		}

		return false
	}

	/**
	 * 清除所有的痕迹（神经元忘记了它的上下文，但连接保持不变）
	 * clears all the traces (the neuron forgets it's context, but the connections remain intact)
	 */
	clear() {
		for (var trace in this.trace.elegibility) {
			this.trace.elegibility[trace] = 0
		}

		for (var trace in this.trace.extended) {
			for (var extended in this.trace.extended[trace]) {
				this.trace.extended[trace][extended] = 0
			}
		}

		this.error.responsibility = this.error.projected = this.error.gated = 0
	}

	/**
	 * 所有的连接都是随机的，痕迹被清除了
	 * all the connections are randomized and the traces are cleared
	 */
	reset() {
		this.clear()

		for (var type in this.connections) {
			for (var connection in this.connections[type]) {
				this.connections[type][connection].weight = Math.random() * .2 - .1
			}
		}

		this.bias = Math.random() * .2 - .1
		this.old = this.state = this.activation = 0
	}

	/**
	 * 将神经元的行为硬编码为优化函数
	 * hardcodes the behaviour of the neuron into an optimized function
	 * @param {*} optimized 
	 * @param {*} layer 
	 */
	optimize(optimized, layer) {

		optimized = optimized || {}
		var store_activation = []
		var store_trace = []
		var store_propagation = []
		var varID = optimized.memory || 0
		var neurons = optimized.neurons || 1
		var inputs = optimized.inputs || []
		var targets = optimized.targets || []
		var outputs = optimized.outputs || []
		var variables = optimized.variables || {}
		var activation_sentences = optimized.activation_sentences || []
		var trace_sentences = optimized.trace_sentences || []
		var propagation_sentences = optimized.propagation_sentences || []
		var layers = optimized.layers || { __count: 0, __neuron: 0 }

		// allocate sentences
		var allocate = function (store) {
			var allocated = layer in layers && store[layers.__count]
			if (!allocated) {
				layers.__count = store.push([]) - 1
				layers[layer] = layers.__count
			}
		};
		allocate(activation_sentences)
		allocate(trace_sentences)
		allocate(propagation_sentences)
		var currentLayer = layers.__count

		// get/reserve space in memory by creating a unique ID for a variablel
		var getVar = function () {
			var args = Array.prototype.slice.call(arguments)

			if (args.length == 1) {
				if (args[0] == 'target') {
					var id = 'target_' + targets.length
					targets.push(varID)
				} else
					var id = args[0]
				if (id in variables)
					return variables[id]
				return variables[id] = {
					value: 0,
					id: varID++
				}
			} else {
				var extended = args.length > 2
				if (extended)
					var value = args.pop()

				var unit = args.shift()
				var prop = args.pop()

				if (!extended)
					var value = unit[prop]

				var id = prop + '_';
				for (var i = 0; i < args.length; i++)
					id += args[i] + '_'
				id += unit.ID
				if (id in variables)
					return variables[id]

				return variables[id] = {
					value: value,
					id: varID++
				}
			}
		}

		// build sentence
		var buildSentence = function () {
			var args = Array.prototype.slice.call(arguments)
			var store = args.pop()
			var sentence = ''
			for (var i = 0; i < args.length; i++)
				if (typeof args[i] == 'string')
					sentence += args[i]
				else
					sentence += 'F[' + args[i].id + ']'

			store.push(sentence + ';')
		}

		// helper to check if an object is empty
		var isEmpty = function (obj) {
			for (var prop in obj) {
				if (obj.hasOwnProperty(prop))
					return false
			}
			return true
		}

		// characteristics of the neuron
		var noProjections = isEmpty(this.connections.projected)
		var noGates = isEmpty(this.connections.gated)
		var isInput = layer == 'input' ? true : isEmpty(this.connections.inputs)
		var isOutput = layer == 'output' ? true : noProjections && noGates

		// optimize neuron's behaviour
		var rate = getVar('rate')
		var activation = getVar(this, 'activation')
		if (isInput)
			inputs.push(activation.id)
		else {
			activation_sentences[currentLayer].push(store_activation)
			trace_sentences[currentLayer].push(store_trace)
			propagation_sentences[currentLayer].push(store_propagation)
			var old = getVar(this, 'old')
			var state = getVar(this, 'state')
			var bias = getVar(this, 'bias')
			if (this.selfconnection.gater)
				var self_gain = getVar(this.selfconnection, 'gain')
			if (this.selfconnected())
				var self_weight = getVar(this.selfconnection, 'weight')
			buildSentence(old, ' = ', state, store_activation)
			if (this.selfconnected())
				if (this.selfconnection.gater)
					buildSentence(state, ' = ', self_gain, ' * ', self_weight, ' * ',
						state, ' + ', bias, store_activation);
				else
					buildSentence(state, ' = ', self_weight, ' * ', state, ' + ',
						bias, store_activation)
			else
				buildSentence(state, ' = ', bias, store_activation)
			for (var i in this.connections.inputs) {
				var input = this.connections.inputs[i]
				var input_activation = getVar(input.from, 'activation')
				var input_weight = getVar(input, 'weight')
				if (input.gater)
					var input_gain = getVar(input, 'gain')
				if (this.connections.inputs[i].gater)
					buildSentence(state, ' += ', input_activation, ' * ',
						input_weight, ' * ', input_gain, store_activation)
				else
					buildSentence(state, ' += ', input_activation, ' * ',
						input_weight, store_activation)
			}
			var derivative = getVar(this, 'derivative')
			switch (this.squash) {
				case Neuron.squash.LOGISTIC:
					buildSentence(activation, ' = (1 / (1 + Math.exp(-', state, ')))',
						store_activation)
					buildSentence(derivative, ' = ', activation, ' * (1 - ',
						activation, ')', store_activation)
					break
				case Neuron.squash.TANH:
					var eP = getVar('aux')
					var eN = getVar('aux_2')
					buildSentence(eP, ' = Math.exp(', state, ')', store_activation);
					buildSentence(eN, ' = 1 / ', eP, store_activation);
					buildSentence(activation, ' = (', eP, ' - ', eN, ') / (', eP, ' + ', eN, ')', store_activation);
					buildSentence(derivative, ' = 1 - (', activation, ' * ', activation, ')', store_activation);
					break;
				case Neuron.squash.IDENTITY:
					buildSentence(activation, ' = ', state, store_activation);
					buildSentence(derivative, ' = 1', store_activation);
					break;
				case Neuron.squash.HLIM:
					buildSentence(activation, ' = +(', state, ' > 0)', store_activation);
					buildSentence(derivative, ' = 1', store_activation);
					break;
				case Neuron.squash.RELU:
					buildSentence(activation, ' = ', state, ' > 0 ? ', state, ' : 0', store_activation);
					buildSentence(derivative, ' = ', state, ' > 0 ? 1 : 0', store_activation);
					break;
			}

			for (var id in this.trace.extended) {
				// calculate extended elegibility traces in advance
				var neuron = this.neighboors[id];
				var influence = getVar('influences[' + neuron.ID + ']');
				var neuron_old = getVar(neuron, 'old');
				var initialized = false;
				if (neuron.selfconnection.gater == this) {
					buildSentence(influence, ' = ', neuron_old, store_trace);
					initialized = true;
				}
				for (var incoming in this.trace.influences[neuron.ID]) {
					var incoming_weight = getVar(this.trace.influences[neuron.ID]
					[incoming], 'weight');
					var incoming_activation = getVar(this.trace.influences[neuron.ID]
					[incoming].from, 'activation');

					if (initialized)
						buildSentence(influence, ' += ', incoming_weight, ' * ', incoming_activation, store_trace);
					else {
						buildSentence(influence, ' = ', incoming_weight, ' * ', incoming_activation, store_trace);
						initialized = true;
					}
				}
			}

			for (var i in this.connections.inputs) {
				var input = this.connections.inputs[i];
				if (input.gater)
					var input_gain = getVar(input, 'gain');
				var input_activation = getVar(input.from, 'activation');
				var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace
					.elegibility[input.ID]);
				if (this.selfconnected()) {
					if (this.selfconnection.gater) {
						if (input.gater)
							buildSentence(trace, ' = ', self_gain, ' * ', self_weight,
								' * ', trace, ' + ', input_gain, ' * ', input_activation,
								store_trace);
						else
							buildSentence(trace, ' = ', self_gain, ' * ', self_weight,
								' * ', trace, ' + ', input_activation, store_trace);
					} else {
						if (input.gater)
							buildSentence(trace, ' = ', self_weight, ' * ', trace, ' + ',
								input_gain, ' * ', input_activation, store_trace);
						else
							buildSentence(trace, ' = ', self_weight, ' * ', trace, ' + ',
								input_activation, store_trace);
					}
				} else {
					if (input.gater)
						buildSentence(trace, ' = ', input_gain, ' * ', input_activation,
							store_trace);
					else
						buildSentence(trace, ' = ', input_activation, store_trace);
				}
				for (var id in this.trace.extended) {
					// extended elegibility trace
					var neuron = this.neighboors[id];
					var influence = getVar('influences[' + neuron.ID + ']');

					var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace
						.elegibility[input.ID]);
					var xtrace = getVar(this, 'trace', 'extended', neuron.ID, input.ID,
						this.trace.extended[neuron.ID][input.ID]);
					if (neuron.selfconnected())
						var neuron_self_weight = getVar(neuron.selfconnection, 'weight');
					if (neuron.selfconnection.gater)
						var neuron_self_gain = getVar(neuron.selfconnection, 'gain');
					if (neuron.selfconnected())
						if (neuron.selfconnection.gater)
							buildSentence(xtrace, ' = ', neuron_self_gain, ' * ',
								neuron_self_weight, ' * ', xtrace, ' + ', derivative, ' * ',
								trace, ' * ', influence, store_trace);
						else
							buildSentence(xtrace, ' = ', neuron_self_weight, ' * ',
								xtrace, ' + ', derivative, ' * ', trace, ' * ',
								influence, store_trace);
					else
						buildSentence(xtrace, ' = ', derivative, ' * ', trace, ' * ',
							influence, store_trace);
				}
			}
			for (var connection in this.connections.gated) {
				var gated_gain = getVar(this.connections.gated[connection], 'gain');
				buildSentence(gated_gain, ' = ', activation, store_activation);
			}
		}
		if (!isInput) {
			var responsibility = getVar(this, 'error', 'responsibility', this.error
				.responsibility);
			if (isOutput) {
				var target = getVar('target');
				buildSentence(responsibility, ' = ', target, ' - ', activation,
					store_propagation);
				for (var id in this.connections.inputs) {
					var input = this.connections.inputs[id];
					var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace
						.elegibility[input.ID]);
					var input_weight = getVar(input, 'weight');
					buildSentence(input_weight, ' += ', rate, ' * (', responsibility,
						' * ', trace, ')', store_propagation);
				}
				outputs.push(activation.id);
			} else {
				if (!noProjections && !noGates) {
					var error = getVar('aux');
					for (var id in this.connections.projected) {
						var connection = this.connections.projected[id];
						var neuron = connection.to;
						var connection_weight = getVar(connection, 'weight');
						var neuron_responsibility = getVar(neuron, 'error',
							'responsibility', neuron.error.responsibility);
						if (connection.gater) {
							var connection_gain = getVar(connection, 'gain');
							buildSentence(error, ' += ', neuron_responsibility, ' * ',
								connection_gain, ' * ', connection_weight,
								store_propagation);
						} else
							buildSentence(error, ' += ', neuron_responsibility, ' * ',
								connection_weight, store_propagation);
					}
					var projected = getVar(this, 'error', 'projected', this.error.projected);
					buildSentence(projected, ' = ', derivative, ' * ', error,
						store_propagation);
					buildSentence(error, ' = 0', store_propagation);
					for (var id in this.trace.extended) {
						var neuron = this.neighboors[id];
						var influence = getVar('aux_2');
						var neuron_old = getVar(neuron, 'old');
						if (neuron.selfconnection.gater == this)
							buildSentence(influence, ' = ', neuron_old, store_propagation);
						else
							buildSentence(influence, ' = 0', store_propagation);
						for (var input in this.trace.influences[neuron.ID]) {
							var connection = this.trace.influences[neuron.ID][input];
							var connection_weight = getVar(connection, 'weight');
							var neuron_activation = getVar(connection.from, 'activation');
							buildSentence(influence, ' += ', connection_weight, ' * ',
								neuron_activation, store_propagation);
						}
						var neuron_responsibility = getVar(neuron, 'error',
							'responsibility', neuron.error.responsibility);
						buildSentence(error, ' += ', neuron_responsibility, ' * ',
							influence, store_propagation);
					}
					var gated = getVar(this, 'error', 'gated', this.error.gated);
					buildSentence(gated, ' = ', derivative, ' * ', error,
						store_propagation);
					buildSentence(responsibility, ' = ', projected, ' + ', gated,
						store_propagation);
					for (var id in this.connections.inputs) {
						var input = this.connections.inputs[id];
						var gradient = getVar('aux');
						var trace = getVar(this, 'trace', 'elegibility', input.ID, this
							.trace.elegibility[input.ID]);
						buildSentence(gradient, ' = ', projected, ' * ', trace,
							store_propagation);
						for (var id in this.trace.extended) {
							var neuron = this.neighboors[id];
							var neuron_responsibility = getVar(neuron, 'error',
								'responsibility', neuron.error.responsibility);
							var xtrace = getVar(this, 'trace', 'extended', neuron.ID,
								input.ID, this.trace.extended[neuron.ID][input.ID]);
							buildSentence(gradient, ' += ', neuron_responsibility, ' * ',
								xtrace, store_propagation);
						}
						var input_weight = getVar(input, 'weight');
						buildSentence(input_weight, ' += ', rate, ' * ', gradient,
							store_propagation);
					}

				} else if (noGates) {
					buildSentence(responsibility, ' = 0', store_propagation);
					for (var id in this.connections.projected) {
						var connection = this.connections.projected[id];
						var neuron = connection.to;
						var connection_weight = getVar(connection, 'weight');
						var neuron_responsibility = getVar(neuron, 'error',
							'responsibility', neuron.error.responsibility);
						if (connection.gater) {
							var connection_gain = getVar(connection, 'gain');
							buildSentence(responsibility, ' += ', neuron_responsibility,
								' * ', connection_gain, ' * ', connection_weight,
								store_propagation);
						} else
							buildSentence(responsibility, ' += ', neuron_responsibility,
								' * ', connection_weight, store_propagation);
					}
					buildSentence(responsibility, ' *= ', derivative,
						store_propagation);
					for (var id in this.connections.inputs) {
						var input = this.connections.inputs[id];
						var trace = getVar(this, 'trace', 'elegibility', input.ID, this
							.trace.elegibility[input.ID]);
						var input_weight = getVar(input, 'weight');
						buildSentence(input_weight, ' += ', rate, ' * (',
							responsibility, ' * ', trace, ')', store_propagation);
					}
				} else if (noProjections) {
					buildSentence(responsibility, ' = 0', store_propagation);
					for (var id in this.trace.extended) {
						var neuron = this.neighboors[id];
						var influence = getVar('aux');
						var neuron_old = getVar(neuron, 'old');
						if (neuron.selfconnection.gater == this)
							buildSentence(influence, ' = ', neuron_old, store_propagation);
						else
							buildSentence(influence, ' = 0', store_propagation);
						for (var input in this.trace.influences[neuron.ID]) {
							var connection = this.trace.influences[neuron.ID][input];
							var connection_weight = getVar(connection, 'weight');
							var neuron_activation = getVar(connection.from, 'activation');
							buildSentence(influence, ' += ', connection_weight, ' * ',
								neuron_activation, store_propagation);
						}
						var neuron_responsibility = getVar(neuron, 'error',
							'responsibility', neuron.error.responsibility);
						buildSentence(responsibility, ' += ', neuron_responsibility,
							' * ', influence, store_propagation);
					}
					buildSentence(responsibility, ' *= ', derivative,
						store_propagation);
					for (var id in this.connections.inputs) {
						var input = this.connections.inputs[id];
						var gradient = getVar('aux');
						buildSentence(gradient, ' = 0', store_propagation);
						for (var id in this.trace.extended) {
							var neuron = this.neighboors[id];
							var neuron_responsibility = getVar(neuron, 'error',
								'responsibility', neuron.error.responsibility);
							var xtrace = getVar(this, 'trace', 'extended', neuron.ID,
								input.ID, this.trace.extended[neuron.ID][input.ID]);
							buildSentence(gradient, ' += ', neuron_responsibility, ' * ',
								xtrace, store_propagation);
						}
						var input_weight = getVar(input, 'weight');
						buildSentence(input_weight, ' += ', rate, ' * ', gradient,
							store_propagation);
					}
				}
			}
			buildSentence(bias, ' += ', rate, ' * ', responsibility,
				store_propagation);
		}
		return {
			memory: varID,
			neurons: neurons + 1,
			inputs: inputs,
			outputs: outputs,
			targets: targets,
			variables: variables,
			activation_sentences: activation_sentences,
			trace_sentences: trace_sentences,
			propagation_sentences: propagation_sentences,
			layers: layers
		}
	}



	static quantity() {
		return {
			neurons: neurons,
			connections: connections
		}
	}
}