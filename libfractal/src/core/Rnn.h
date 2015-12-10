/*
   Copyright 2015 Kyuyeon Hwang (kyuyeon.hwang@gmail.com)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/


#ifndef FRACTAL_RNN_H_
#define FRACTAL_RNN_H_

#include <unordered_map>
#include <unordered_set>
#include <list>
#include <stack>
#include <string>
#include <ostream>
#include <vector>

#include "InitWeightParam.h"
#include "Engine.h"
#include "Layer.h"
#include "Probe.h"
#include "Connection.h"
#include "FractalCommon.h"


/// The topmost namespace of libfractal.
/*! Contains all libfractal classes. */
namespace fractal
{

/// A network structure container.
/*! This class contains a neural network structure with graph-based representation.
 *  Nodes and edges correspond to layers and connections respectively.
 */
class Rnn
{
public:

    /// The constructor.
    /*! Initialize variables.
     */
    Rnn();

    /// The destructor.
    /*! Free resources.
     */
    virtual ~Rnn();

    /// Set a computation engine.
    /*! If \a engine is set to \c NULL, free resources.
     *  \param engine A pointer to an Engine instance.
     */
    void SetEngine(Engine *engine);

    /// Get the computation engine.
    /*! \return A pointer to the Engine instance of this RNN.
     *      The return value is \c NULL when the engine is not set.
     */
    Engine *GetEngine() const;

    /// Add a layer.
    /*! \param name %Layer name.
     *  \param actType Activation function type.
     *  \param aggType Aggregation function type.
     *  \param size %Layer size.
     *  \param param Extra layer parameter.
     */
    void AddLayer(const std::string &name, ActType actType, AggType aggType, const unsigned long size, const LayerParam &param = LayerParam());

    /// Add a connection.
    /*! \param from The name of the anterior layer.
     *  \param to The name of the posterior layer.
     *  \param param Extra connection parameter.
     */
    void AddConnection(const std::string &from, const std::string &to, const ConnParam &param = ConnParam());

    /// Delete a Layer.
    /*! All connections from or to this layer is automatically removed.
     *  \param name The name of the layer to be deleted.
     */
    void DeleteLayer(const std::string &name);

    /// Delete a connection.
    /*! \param from The name of the anterior layer of the connection to be deleted.
     *  \param to The name of the posterior layer of the connection to be deleted.
     */
    void DeleteConnection(const std::string &from, const std::string &to);

    /// Link a probe to a layer.
    /*! \param probe The probe instance.
     *  \param layerName %Layer name.
     */
    void LinkProbe(Probe &probe, const std::string &layerName);

    /// Set the mini-batch size.
    /*! Unroll the network \a nUnroll times with a ring buffer.
     *  There are \a nStream independent streams.
     *  \param nStream The number of streams.
     *  \param nUnroll The amount of unrolling.
     */
    void SetBatchSize(const unsigned long nStream, const unsigned long nUnroll);

    /// Get the number of independent streams.
    /*! \return The number of streams.
     */
    const unsigned long GetNumStreams() const;

    /// Get the amount of the network unrolling.
    /*! \return The amount of unrolling.
     */
    const unsigned long GetNumUnrollSteps() const;

    /// Initialize layer activations of entire streams for forward propagation.
    /*! \param idxFrom The start index of the ring buffer.
     *  \param idxTo The end index of the ring buffer.
     */
    void InitForward(const unsigned long idxFrom, const unsigned long idxTo);

    /// Initialize errors in connections of entire streams for backward propagation.
    /*! \param idxFrom The start index of the ring buffer.
     *  \param idxTo The end index of the ring buffer.
     */
    void InitBackward(const unsigned long idxFrom, const unsigned long idxTo);

    /// Initialize all weights.
    /*! \param param Weight initialization parameter.
     */
    void InitWeights(const InitWeightParam &param);

    /// Initialize AdaDelta.
    /*! Initialize mean-square variables.
     *  \param decayRate Decay rate of exponential averaging.
     *  \param initDenominator Set to \c true to initialize the denominator mean-square variable.
     */
    void InitAdadelta(const FLOAT decayRate, const bool initDenominator = true);

    /// Initialize Nesterov momentum.
    /*! Initialize the velocities to zero.
     */
    void InitNesterov();

    /// Initialize RMSprop.
    /*! Initialize the mean-square variables to one.
     *  \param decayRate Decay rate of exponential averaging.
     */
    void InitRmsprop(const FLOAT decayRate);

    /// Forward propagation.
    /*! Perform forward propagation from \a idxFrom to \a idxTo.
     *  The total number of forward steps per stream is
     *  (\a idxTo - \a idxFrom + 1).
     *  \param idxFrom The start index of the ring buffer.
     *  \param idxTo The end index of the ring buffer.
     */
    void Forward(const unsigned long idxFrom, const unsigned long idxTo);

    /// Backward propagation.
    /*! Perform backward propagation from \a idxFrom to \a idxTo.
     *  The total number of backward steps per stream is
     *  (\a idxTo - \a idxFrom + 1).
     *  \param idxFrom The start index of the ring buffer.
     *  \param idxTo The end index of the ring buffer.
     */
    void Backward(const unsigned long idxFrom, const unsigned long idxTo);

    /// Compute the derivatives of the activation functions with respect to the layer states.
    /*! \param idxFrom The start index of the ring buffer.
     *  \param idxTo The end index of the ring buffer.
     */
    void CalcActDeriv(const unsigned long idxFrom, const unsigned long idxTo);

    /// Update the weights.
    /*! \param idxFrom The start index of the ring buffer.
     *  \param idxTo The end index of the ring buffer.
     *  \param rate Learning rate.
     *  \param momentum Momentum.
     *  \param adadelta Set to \c true to perform AdaDelta.
     *  \param rmsprop Set to \c true to perform RMSprop.
     *  \note \a AdaDelta and \a RMSprop cannot be set to \c true at the same time.
     */
    void UpdateWeights(const unsigned long idxFrom, const unsigned long idxTo,
            const FLOAT rate, const FLOAT momentum, const bool adadelta, const bool rmsprop);

    /// Enable dropout.
    /*! \param enable Set to \c true to enable dropout.
     */
    void EnableDropout(const bool enable);

    /// Generate dropout mask.
    /*! \param idxFrom The start index of the ring buffer.
     *  \param idxTo The end index of the ring buffer.
     */
    void GenerateDropoutMask(const unsigned long idxFrom, const unsigned long idxTo);

    /// Process forward and backard weights.
    /*! \param noise Standard deviation of noise to be added to forward and backward weights.
     */
    void ProcessWeights(const FLOAT noise);

    /// Fix current network weights.
    /*! Current weights will not be updated during training.
     *  Newly added parameters after calling this function will not be affected.
     *  /param enable Set to \c true to fix or \false to unfix weights.
     */
    void FixCurrentWeights(const bool enable);

    /// Wait until all asynchronous operations are finished.
    void Synchronize();

    /// Force a \a stream to wait until all asynchronous operations of this RNN are finished.
    /*! \param stream PStream object.
     */
    void StreamWait(PStream &stream);

    /// Get ready to perform forward and backward propagation.
    /*! Analyze the network structure to find strongly connected components (loops)
     *  and determine the activation order. Automatically called when needed.
     */
    void Ready();

    /// Print the network in the forward activation order.
    /*! \param outStream Output stream.
     */
    void PrintNetwork(std::ostream &outStream);

    /// Free resources.
    void Clear();

    /// Save the current network states.
    /*! \param path Path to save.
     */
    void SaveState(const std::string &path);

    /// Load the previously saved network states.
    /*! \param path Path to load.
     */
    void LoadState(const std::string &path);

    /// Get the total number of weights.
    const unsigned long GetNumWeights();

    /// Set which compute locations of the linked engine are used.
    /*! \param computeLoc Vector of compute locations to enable.
     */
    void SetComputeLocs(const std::vector<unsigned long> &computeLoc);

    /// Get which compute locations of the linked engine are used.
    /*! \param computeLoc Vector of compute locations.
     */
    std::vector<unsigned long > &GetComputeLocs();

protected:

    /// Structure for strongly connected components.
    typedef std::list<Layer *> Scc;

    /// List structure of Layer objects.
    typedef std::list<Layer *> LayerList;

    /// Map structure that maps each layer name to the corresponding Layer object.
    typedef std::unordered_map<std::string, Layer *> LayerMap;

    /// Set structure that contains the pointers to all connections.
    typedef std::unordered_set<Connection *> ConnSet;

    /// List structure of Scc objects.
    typedef std::list<Scc *> SccList;

    /// List structure of PStream objects.
    typedef std::list<PStream *> PStreamList;

    /// Find a layer using its name.
    /*! \param layerName %Layer name.
     *  \return The pointer to the corresponding layer.
     *      If there is no matching layer, \c NULL is returned.
     */
    Layer *FindLayer(const std::string &layerName);

    /// Add a connection.
    /*! \param from The pointer to the anterior layer.
     *  \param to The pointer to the posterior layer.
     *  \param param Extra connection parameter.
     */
    void AddConnection(Layer *const from, Layer *const to, const ConnParam &param);

    /// Delete a connection.
    /*! \param from The pointer to the anterior layer of the connection to be deleted.
     *  \param to The pointer to the posterior layer of the connection to be deleted.
     */
    void DeleteConnection(Layer *const from, Layer *const to);

    /// Link a probe to a layer.
    /*! \param probe The probe instance.
     *  \param layer The pointer to the layer.
     */
    void LinkProbe(Probe &probe, Layer *const layer);

    /// Clear all layers.
    void ClearLayers();

    /// Clear all connections.
    void ClearConnections();

    /// Clear all SCCs.
    void ClearSccList();

    /// Clear all PStreams.
    void ClearPStreams();

    /// Perform Tarjan's strongly connected component (SCC) algorithm.
    /*! This function analyzes the network structure to find SCCs and loops.
     */
    void Tarjan();

    /// Create an SCC and perform topological sort.
    /*! Subroutine of Tarjan().
     *  \param sccStack Stack of nodes (Layer objects).
     *  \param root The pointer to the root node (Layer object).
     *  \param group The group index or SCC index.
     *  \return The pointer to the newly created SCC object.
     *  \note Memory space is dynamically allocated for the returned SCC object.
     */
    Scc *const CreateScc(std::stack<Layer *> &sccStack, const Layer *const root, const long group);

    /// Create PStream objects.
    void CreatePStreams();

    /// Create the default PStream object with the location \a loc.
    /*! \param loc Location.
     */
    void CreateDefaultPStream(const unsigned long loc);

    /// Destroy the default PStream object.
    void DestroyDefaultPStream();

    /// Engine pointer.
    Engine *engine;

    /// Layer map.
    LayerMap layerMap;

    /// Connection set.
    ConnSet connSet;

    /// Scc list.
    SccList sccList;

    /// PStream list.
    PStreamList pStreamList;

    /// Default PStream.
    PStream *defaultPStream;

    /// The number of independent streams.
    unsigned long nStream;

    /// Unrolling amount.
    /*! Each stream is unrolled \a nUnroll steps.
     */
    unsigned long nUnroll;

    /// Vector of enabled compute locations.
    std::vector<unsigned long> computeLoc;

    /// Indicates whether the network is analyzed or not.
    bool isReady;
};

}

#endif /* FRACTAL_RNN_H_ */

