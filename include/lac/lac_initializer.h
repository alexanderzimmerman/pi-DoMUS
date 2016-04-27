#ifndef _LAC_INITIALIZER_H_
#define _LAC_INITIALIZER_H_

// This includes all types we know of.
#include "lac/lac_type.h"
#include <mpi.h>

/**
 * General class, used to initialize different types of Vectors, Matrices and
 * Sparsity Patterns using a common interface.
 */
class ScopedLACInitializer
{
public:
  ScopedLACInitializer(const std::vector<types::global_dof_index> &dofs_per_block,
                       const std::vector<IndexSet> &owned,
                       const std::vector<IndexSet> &relevant,
                       const MPI_Comm &comm = MPI_COMM_WORLD):
    dofs_per_block(dofs_per_block),
    owned(owned),
    relevant(relevant),
    comm(comm)
  {};

  /**
   * Initialize a non ghosted TrilinosWrappers::MPI::BlockVector.
   */
  void operator() (TrilinosWrappers::MPI::BlockVector &v, bool fast=false)
  {
    v.reinit(owned, comm, fast);
  };


  /**
   * Initialize a ghosted TrilinosWrappers::MPI::BlockVector.
   */
  void ghosted(TrilinosWrappers::MPI::BlockVector &v, bool fast=false)
  {
    v.reinit(owned, relevant, comm, fast);
  };

  /**
   * Initialize a serial BlockVector<double>.
   */
  void operator() (BlockVector<double> &v, bool fast=false)
  {
    v.reinit(dofs_per_block, fast);
  };


  /**
   * Initiale a ghosted BlockVector<double>. Will throw an exception and
   * should not be called...
   */
  void ghosted(BlockVector<double> &, bool fast=false)
  {
    Assert(false, ExcInternalError("You tried to create a ghosted vector in a serial run."));
    (void)fast;
  };

  /**
   * Initialize a Trilinos Block Sparse Matrix.
   */
  template<int dim, int spacedim>
  void operator() (TrilinosWrappers::BlockSparseMatrix &matrix,
                   const DoFHandler<dim, spacedim> &dh,
                   const ConstraintMatrix &cm,
                   const Table<2,DoFTools::Coupling> &coupling)
  {
    TrilinosWrappers::BlockSparsityPattern sp(owned,
                                              owned,
                                              relevant,
                                              comm);

    DoFTools::make_sparsity_pattern (dh,
                                     coupling, sp,
                                     cm, false,
                                     Utilities::MPI::this_mpi_process(comm));
    sp.compress();

    matrix.reinit(sp);
  }

  /**
   * Initialize a Deal.II Block Sparse Matrix.
   */
  template<int dim, int spacedim>
  void operator() (dealii::BlockSparseMatrix<double> &matrix,
                   const DoFHandler<dim, spacedim> &dh,
                   const ConstraintMatrix &cm,
                   const Table<2,DoFTools::Coupling> &coupling)
  {
    dealii::BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

    DoFTools::make_sparsity_pattern (dh,
                                     coupling, dsp,
                                     cm, false);
    dsp.compress();

    dealii::BlockSparsityPattern sp;

    sp.copy_from(dsp);

    matrix.reinit(sp);
  }

private:
  /**
   * Dofs per block.
   */
  const std::vector<types::global_dof_index> &dofs_per_block;

  /**
   * Owned dofs per block.
   */
  const std::vector<IndexSet> &owned;

  /**
   * Relevant dofs per block.
   */
  const std::vector<IndexSet> &relevant;

  /**
   * MPI Communicator.
   */
  const MPI_Comm &comm;
};


// A few useful functions
inline void compress(TrilinosWrappers::BlockSparseMatrix &m,
                     VectorOperation::values op)
{
  m.compress(op);
}

inline void compress(dealii::BlockSparseMatrix<double> &,
                     VectorOperation::values )
{
}
#endif
