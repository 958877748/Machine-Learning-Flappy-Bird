import Neuron from "./Neuron"

export let connections = 0

/**
 * 2个神经元之间的连接
 */
export default class Connection {
	/**
	 * 
	 * @param {Neuron} from 输入神经元
	 * @param {Neuron} to 输出神经元
	 * @param {number} weight 权重
	 */
	constructor(from, to, weight) {
		if (!from || !to)
			throw new Error("Connection Error: Invalid neurons")

		/**
		 * 唯一id
		 */
		this.ID = Connection.uid()
		/**
		 * 输入神经元
		 */
		this.from = from
		/**
		 * 输出神经元
		 */
		this.to = to
		/**
		 * 权重
		 */
		this.weight = typeof weight == 'undefined' ? Math.random() * .2 - .1 : weight
		this.gain = 1
		this.gater = null
	}

	static uid() {
		return connections++
	}
}