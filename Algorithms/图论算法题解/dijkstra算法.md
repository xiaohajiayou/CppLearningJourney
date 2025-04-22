前置知识

阅读本文前，你需要先学习：

*   [图结构基础及通用实现](https://labuladong.online/algo/data-structure-basic/graph-basic/)
*   [二叉树的 DFS/BFS 遍历](https://labuladong.online/algo/data-structure-basic/binary-tree-traverse-basic/)
*   [图结构的 DFS/BFS 遍历](https://labuladong.online/algo/data-structure-basic/graph-traverse-basic/)

一句话总结

Dijkstra 算法是一种用于计算图中单源最短路径的算法，本质上是一个经过特殊改造的 BFS 算法，改造点有两个：

1、使用 [优先级队列](https://labuladong.online/algo/data-structure-basic/binary-heap-implement/)，而不是普通队列进行 BFS 算法。

2、添加了一个备忘录，记录起点到每个可达节点的最短路径权重和。

学习 Dijkstra 最短路径算法之前，你需要先了解 [图结构基础及通用代码实现](https://labuladong.online/algo/data-structure-basic/graph-basic/)，下面的讲解中，我会用到图结构 `Graph` 的通用 API。

另外，你必须要理解 [二叉树的 DFS/BFS 遍历](https://labuladong.online/algo/data-structure-basic/binary-tree-traverse-basic/) 以及 [图结构的 DFS/BFS 遍历](https://labuladong.online/algo/data-structure-basic/graph-traverse-basic/) 中 BFS 遍历的基本原理，因为 Dijkstra 算法本质上就是一个经过特殊改造的 BFS 算法。

在讲解二叉树和图结构的 BFS 遍历算法时，我同时给出了三种 BFS 算法的写法，如果忘了可以回去复习一下。

其中第三种 BFS 算法相对复杂一些，但是最灵活，因为它新建了一个 `State` 类，允许每个节点独立维护一些额外信息。

具体代码如下：


cpp 🤖

    // 多叉树的层序遍历
    // 每个节点自行维护 State 类，记录深度等信息
    class State {
    public:
        Node* node;
        int depth;
    
        State(Node* node, int depth) : node(node), depth(depth) {}
    };
    
    void levelOrderTraverse(Node* root) {
        if (root == nullptr) {
            return;
        }
        queue<State> q;
        // 记录当前遍历到的层数（根节点视为第 1 层）
        q.push(State(root, 1));
    
        while (!q.empty()) {
            State state = q.front();
            q.pop();
            Node* cur = state.node;
            int depth = state.depth;
            // 访问 cur 节点，同时知道它所在的层数
            cout << "depth = " << depth << ", val = " << cur->val << endl;
    
            for (Node* child : cur->children) {
                q.push(State(child, depth + 1));
            }
        }
    }
    
    
    // 图结构的 BFS 遍历，从节点 s 开始进行 BFS，且记录路径的权重和
    // 每个节点自行维护 State 类，记录从 s 走来的权重和
    class State {
    public:
        // 当前节点 ID
        int node;
        // 从起点 s 到当前节点的权重和
        int weight;
    
        State(int node, int weight) : node(node), weight(weight) {}
    };
    
    void bfs(const Graph& graph, int s) {
        vector<bool> visited(graph.size(), false);
        queue<State> q;
    
        q.push(State(s, 0));
        visited[s] = true;
    
        while (!q.empty()) {
            State state = q.front();
            q.pop();
            int cur = state.node;
            int weight = state.weight;
            cout << "visit " << cur << " with path weight " << weight << endl;
            for (const Edge& e : graph.neighbors(cur)) {
                if (!visited[e.to]) {
                    q.push(State(e.to, weight + e.weight));
                    visited[e.to] = true;
                }
            }
        }
    }


这种写法对于树结构来说有些多此一举，但是对于加权图来说，就非常有用了。

在这个可视化面板中，我创建了一幅加权图。你可以多次点击 `console.log` 这一行代码，注意命令行的输出，这种写法可以在遍历节点的同时得知起点到当前节点的路径和：

我们即将实现的 Dijkstra 算法就是基于这个算法的改进，每个节点都需要记录从起点到自己的最短路径权重和，再结合 [优先级队列](https://labuladong.online/algo/data-structure-basic/binary-heap-implement/) 这种能够动态排序的数据结构，就可以高效地计算出最短路径了。

下面来具体介绍 Dijkstra 算法的通用代码实现。


[Dijkstra 函数签名](#)
------------------

首先，我们可以写一个 Dijkstra 算法的通用函数签名：


cpp 🤖

    // 输入一幅图和一个起点 start，计算 start 到其他节点的最短距离
    vector<int> dijkstra(int start, Graph graph);


输入是一幅图 `graph` 和一个起点 `start`，返回是一个记录最短路径权重的数组，比方下面这个例子：

    int[] distTo = dijkstra(3, graph);

`distTo` 数组中存储节点 `3` 作为起点到其他节点的最小路径和，比如从起点 `3` 到节点 `6` 的最短路径权重和就是 `distTo[6]`。

因为是本质上就是 BFS 嘛，所以标准的 Dijkstra 算法会从起点 `start` 开始遍历，把到所有其他可达节点的最短路径都算出来。

当然，如果你的需求只是计算从起点 `start` 到某一个终点 `end` 的最短路径，那么在标准 Dijkstra 算法上稍作修改就可以更高效地完成这个需求，这个我们后面再说。

[`State` 类](#)
--------------

我们也需要一个 `State` 类来辅助 BFS 算法的运行，清晰起见，我们用 `id` 变量记录当前节点 ID，用 `distFromStart` 变量记录从起点到当前节点的距离。


cpp 🤖

    class State {
        // 图节点的 id
        int id;
        // 从 start 节点到当前节点的距离
        int distFromStart;
    
    public:
        State(int id, int distFromStart) {
            this->id = id;
            this->distFromStart = distFromStart;
        }
    };


[`distTo` 记录最短路径](#)
--------------------

加权图中的 Dijkstra 算法和无权图中的普通 BFS 算法不同，在 Dijkstra 算法中，你第一次经过某个节点时的路径权重，不见得就是最小的，所以对于同一个节点，我们可能会经过多次，而且每次的 `distFromStart` 可能都不一样，比如下图：

![](https://labuladong.online/algo/images/dijkstra/3.jpeg)

我会经过节点 `5` 三次，每次的 `distFromStart` 值都不一样，那我取 `distFromStart` 最小的那次，不就是从起点 `start` 到节点 `5` 的最短路径权重了么？

所以我们需要一个 `distTo` 数组来记录从起点 `start` 到每个节点的最短路径权重和，起到一个备忘录的作用。

当重复遍历到同一个节点时，我们可以比较一下当前的 `distFromStart` 和 `distTo` 中的值，如果当前的更小，就更新 `distTo`，反之，就不用再往后继续遍历了。

[代码实现](#)
---------

Dijkstra 的伪码逻辑如下：


cpp 🤖

    // 输入一幅图和一个起点 start，计算 start 到其他节点的最短距离
    vector<int> dijkstra(int start, Graph* graph) {
        // 图中节点的个数
        int V = graph.size();
        // 记录最短路径的权重，你可以理解为 dp table
        // 定义：distTo[i] 的值就是节点 start 到达节点 i 的最短路径权重
        int distTo[V];
        // 求最小值，所以 dp table 初始化为正无穷
        memset(distTo, INT_MAX, sizeof(distTo));
        // base case，start 到 start 的最短距离就是 0
        distTo[start] = 0;
    
        // 优先级队列，distFromStart 较小的排在前面
        priority_queue<State, vector<State>, decltype(&comparator)> pq(&comparator);
    
        // 从起点 start 开始进行 BFS
        pq.push(State(start, 0));
    
        while (!pq.empty()) {
            State curState = pq.top();
            pq.pop();
            int curNodeID = curState.id;
            int curDistFromStart = curState.distFromStart;
    
            if (curDistFromStart > distTo[curNodeID]) {
                // 已经有一条更短的路径到达 curNode 节点了
                continue;
            }
            // 将 curNode 的相邻节点装入队列
            for (int nextNodeID: graph.neighbors(curNodeID)) {
                // 看看从 curNode 达到 nextNode 的距离是否会更短
                int distToNextNode = distTo[curNodeID] + graph.weight(curNodeID, nextNodeID);
                if (distTo[nextNodeID] > distToNextNode) {
                    // 更新 dp table
                    distTo[nextNodeID] = distToNextNode;
                    // 将这个节点以及距离放入队列
                    pq.push(State(nextNodeID, distToNextNode));
                }
            }
        }
        vector<int> result;
        for (int i = 0; i < V; i++) {
            result.push_back(distTo[i]);
        }
        return result;
    }


对比普通的 BFS 算法，你可能会有以下疑问：

**1、没有 `visited` 集合记录已访问的节点，所以一个节点会被访问多次，会被多次加入队列，那会不会导致队列永远不为空，造成死循环**？

**2、为什么用优先级队列 `PriorityQueue` 而不是 `LinkedList` 实现的普通队列？为什么要按照 `distFromStart` 的值来排序**？

**3、如果我只想计算起点 `start` 到某一个终点 `end` 的最短路径，是否可以修改算法，提升一些效率**？

我们先回答第一个问题，为什么这个算法不用 `visited` 集合也不会死循环。

对于这类问题，我教你一个思考方法：

循环结束的条件是队列为空，那么你就要注意看什么时候往队列里放元素（调用 `offer` 方法），再注意看什么时候从队列往外拿元素（调用 `poll` 方法）。

`while` 循环每执行一次，都会往外拿一个元素，但想往队列里放元素，可就有很多限制了，必须满足下面这个条件：


cpp 🤖

    // 看看从 curNode 达到 nextNode 的距离是否会更短
    if (distTo.at(nextNodeID) > distToNextNode) {
        // 更新 dp table
        distTo[nextNodeID] = distToNextNode;
        pq.push(State(nextNodeID, distToNextNode));
    }


这也是为什么我说 `distTo` 数组可以理解成我们熟悉的 dp table，因为这个算法逻辑就是在不断的最小化 `distTo` 数组中的元素：

如果你能让到达 `nextNodeID` 的距离更短，那就更新 `distTo[nextNodeID]` 的值，让你入队，否则的话对不起，不让入队。

**因为两个节点之间的最短距离（路径权重）肯定是一个确定的值，不可能无限减小下去，所以队列一定会空，队列空了之后，`distTo` 数组中记录的就是从 `start` 到其他节点的「最短距离」**。

接下来解答第二个问题，为什么要用 `PriorityQueue` 而不是 `LinkedList` 实现的普通队列？

如果你非要用普通队列，其实也没问题的，你可以直接把 `PriorityQueue` 改成 `LinkedList`，也能得到正确答案，但是效率会低很多。

**Dijkstra 算法使用优先级队列，主要是为了效率上的优化，类似一种贪心算法的思路**。

为什么说是一种贪心思路呢，比如说下面这种情况，你想计算从起点 `start` 到终点 `end` 的最短路径权重：

![](https://labuladong.online/algo/images/dijkstra/4.jpeg)

假设你当前只遍历了图中的这几个节点，那么你下一步准备遍历那个节点？这三条路径都可能成为最短路径的一部分，**但你觉得哪条路径更有「潜力」成为最短路径中的一部分**？

从目前的情况来看，显然橙色路径的可能性更大嘛，所以我们希望节点 `2` 排在队列靠前的位置，优先被拿出来向后遍历。

所以我们使用 `PriorityQueue` 作为队列，让 `distFromStart` 的值较小的节点排在前面，这就类似我们之前讲 [贪心算法](https://labuladong.online/algo/essential-technique/greedy/) 说到的贪心思路，可以很大程度上优化算法的效率。

大家应该听过 Bellman-Ford 算法，这个算法是一种更通用的最短路径算法，因为它可以处理带有负权重边的图，Bellman-Ford 算法逻辑和 Dijkstra 算法非常类似，用到的就是普通队列，本文就提一句，后面有空再具体写。

接下来说第三个问题，如果只关心起点 `start` 到某一个终点 `end` 的最短路径，是否可以修改代码提升算法效率。

肯定可以的，因为我们标准 Dijkstra 算法会算出 `start` 到所有其他节点的最短路径，你只想计算到 `end` 的最短路径，相当于减少计算量，当然可以提升效率。

需要在代码中做的修改也非常少，只要改改函数签名，再加个 if 判断就行了：


cpp 🤖

    // 输入起点 start 和终点 end，计算起点到终点的最短距离
    int dijkstra(int start, int end, vector<int>* graph) {
    
        // ...
    
        while (!pq.empty()) {
            State curState = pq.top();
            pq.pop();
            int curNodeID = curState.id;
            int curDistFromStart = curState.distFromStart;
    
            // 在这里加一个判断就行了，其他代码不用改
            if (curNodeID == end) {
                return curDistFromStart;
            }
    
            if (curDistFromStart > distTo[curNodeID]) {
                continue;
            }
    
            // ...
        }
    
        // 如果运行到这里，说明从 start 无法走到 end
        return INT_MAX;
    }


因为优先级队列自动排序的性质，**每次**从队列里面拿出来的都是 `distFromStart` 值最小的，所以当你**第一次**从队列中拿出终点 `end` 时，此时的 `distFromStart` 对应的值就是从 `start` 到 `end` 的最短距离。

这个算法较之前的实现提前 return 了，所以效率有一定的提高。

这是 Dijkstra 算法的可视化面板，你可以点击其中的代码，查看算法的执行过程：

[Algorithm visualize](https://labuladong.online/algo-visualize/tutorial/dijkstra-example/)   **算法可视化面板**Link copied!

[时间复杂度分析](#)
------------

Dijkstra 算法的时间复杂度是多少？你去网上查，可能会告诉你是 O(ElogV)O(ElogV)O(ElogV)，其中 `E` 代表图中边的条数，`V` 代表图中节点的个数。

因为理想情况下优先级队列中最多装 `V` 个节点，对优先级队列的操作次数和 `E` 成正比，所以整体的时间复杂度就是 O(ElogV)O(ElogV)O(ElogV)。

不过这是理想情况，Dijkstra 算法的代码实现有很多版本，不同编程语言或者不同数据结构 API 都会导致算法的时间复杂度发生一些改变。

比如本文实现的 Dijkstra 算法，使用了 Java 的 `PriorityQueue` 这个数据结构，这个容器类底层使用二叉堆实现，但没有提供通过索引操作队列中元素的 API，所以队列中会有重复的节点，最多可能有 `E` 个节点存在队列中。

所以本文实现的 Dijkstra 算法复杂度并不是理想情况下的 O(ElogV)O(ElogV)O(ElogV)，而是 O(ElogE)O(ElogE)O(ElogE)，可能会略大一些，因为图中边的条数一般是大于节点的个数的。

不过就对数函数来说，就算真数大一些，对数函数的结果也大不了多少，所以这个算法实现的实际运行效率也是很高的，以上只是理论层面的时间复杂度分析，供大家参考。

在下一节 [Dijkstra 算法习题](https://labuladong.online/algo/problem-set/dijkstra/) 中，我们会用 Dijkstra 算法解决一些具体的算法问题。

* * *

**引用本文的题目**


[1514. 概率最大的路径](https://leetcode.cn/problems/path-with-maximum-probability/?show=1) 

[1631. 最小体力消耗路径](https://leetcode.cn/problems/path-with-minimum-effort/?show=1) 

[286. 墙与门](https://leetcode.cn/problems/walls-and-gates/?show=1) 🔒

[310. 最小高度树](https://leetcode.cn/problems/minimum-height-trees/?show=1) 

[329. 矩阵中的最长递增路径](https://leetcode.cn/problems/longest-increasing-path-in-a-matrix/?show=1) 🔴

[505. 迷宫 II](https://leetcode.cn/problems/the-maze-ii/?show=1) 🔒

[542. 01 矩阵](https://leetcode.cn/problems/01-matrix/?show=1) 

[743. 网络延迟时间](https://leetcode.cn/problems/network-delay-time/?show=1) 

