/****************************************************************************
<one line to give the program's name and a brief idea of what it does.>
Copyright (C) <2021>  <JiHuaCao>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

*****************************************************************************/
#ifndef PROJECT_BASE_ALGORITHM_TREE_H
#define PROJECT_BASE_ALGORITHM_TREE_H
#include <cassert>
#include <memory>
#include <stdexcept>
#include <iterator>
#include <set>
#include <queue>
#include <algorithm>
#include <cstddef>

#include <ProjectBase/basic_algorithm/Define.hpp>

namespace ProjectBase{
    namespace Tree{
		/**
		 * \brief brief
		 * \note note
		 * \author none
		 * \since version
		 * */
		template<class T>
		class _tree_node {
			public:
				_tree_node();
				_tree_node(const T&);
				_tree_node(T&&);
				T data;
			public:
				virtual _tree_node<T>* pre() const=0;
				virtual _tree_node<T>* next() const=0;
				virtual _tree_node<T>* first_child() const=0;
				virtual _tree_node<T>* last_child() const=0;
				virtual _tree_node<T>* parent() const=0;
		};

		template<class T>
		_tree_node<T>::_tree_node(){

		};

		template<class T>
		_tree_node<T>::_tree_node(const T& val)
		: data(val){

		};

		template<class T>
		_tree_node<T>::_tree_node(T&& val)
		: data(val){

		};

		/**
		 * \brief brief
		 * \note note
		 * \author none
		 * \since version
		 * */
		template<class T>
		class _tree{

		};

		/// A node in the tree, combining links to other nodes as well as the actual data.
		/**
		 * \brief duplex_tree_node，双工链表构成的树节点，有父子节点链表、兄弟节点链表
		 * \note 空间复杂度比较高，但是时间复杂度比较低
		 * \author none
		 * \since v0.0.1
		 * */
		template<class T>
		class duplex_tree_node : public _tree_node<T> { // size: 5*4=20 bytes (on 32 bit arch), can be reduced by 8.
			public:
				duplex_tree_node();
				duplex_tree_node(const T&);
				duplex_tree_node(T&&);

				duplex_tree_node<T> *_parent; // 指针记录父节点
				duplex_tree_node<T> *_first_child, *_last_child; // 链表方式记录子节点指针
				duplex_tree_node<T> *_prev_sibling, *_next_sibling; // 链表的方式记录兄弟节点指针
			public:
				_tree_node<T>* pre() const;
				_tree_node<T>* next() const;
				_tree_node<T>* first_child() const;
				_tree_node<T>* last_child() const;
				_tree_node<T>* parent() const;
		}; 

		template<class T>
		duplex_tree_node<T>::duplex_tree_node()
			: _tree_node<T>(), _parent(0), _first_child(0), _last_child(0), _prev_sibling(0), _next_sibling(0)
			{
			}

		template<class T>
		duplex_tree_node<T>::duplex_tree_node(const T& val)
			: _tree_node<T>(val), _parent(0), _first_child(0), _last_child(0), _prev_sibling(0), _next_sibling(0){
		}

		template<class T>
		duplex_tree_node<T>::duplex_tree_node(T&& val)
			: _tree_node<T>(val), _parent(0), _first_child(0), _last_child(0), _prev_sibling(0), _next_sibling(0){
		}

		template<class T>
		_tree_node<T>* duplex_tree_node<T>::pre() const{
			return _prev_sibling;
		}

		template<class T> 
		_tree_node<T>* duplex_tree_node<T>::next() const{
			return _next_sibling;
		}

		template<class T> 
		_tree_node<T>* duplex_tree_node<T>::first_child() const{
			return _first_child;
		}

		template<class T> 
		_tree_node<T>* duplex_tree_node<T>::last_child() const{
			return _last_child;
		}

		template<class T> 
		_tree_node<T>* duplex_tree_node<T>::parent() const{
			return _parent;
		}

		class navigation_error : public std::logic_error {
			public:
				navigation_error(const std::string& s) : std::logic_error(s)
					{
		//			assert(1==0);
		//			std::ostringstream str;
		//			std::cerr << boost::stacktrace::stacktrace() << std::endl;
		//			str << boost::stacktrace::stacktrace();
		//			stacktrace=str.str();
					};

		//		virtual const char *what() const noexcept override
		//			{
		//			return (std::logic_error::what()+std::string("; ")+stacktrace).c_str();
		//			}
		//
		//		std::string stacktrace;
		};

		template <class T, class tree_nodeallocator = std::allocator<duplex_tree_node<T> > >
		class tree {
			protected:
				typedef duplex_tree_node<T> tree_node;
			public:
				/// Value of the data stored at a node.
				typedef T value_type;

				class iterator_base;
				class pre_order_iterator;
				class post_order_iterator;
				class sibling_iterator;
		      	class leaf_iterator;

				tree();                                         // empty constructor
				tree(const T&);                                 // constructor setting given element as head
				tree(const iterator_base&);
				tree(const tree<T, tree_nodeallocator>&);      // copy constructor
				tree(tree<T, tree_nodeallocator>&&);           // move constructor
				~tree();
				tree<T,tree_nodeallocator>& operator=(const tree<T, tree_nodeallocator>&);   // copy assignment
				tree<T,tree_nodeallocator>& operator=(tree<T, tree_nodeallocator>&&);        // move assignment

		      /// Base class for iterators, only pointers stored, no traversal logic.
		#ifdef __SGI_STL_PORT
				class iterator_base : public stlport::bidirectional_iterator<T, ptrdiff_t> {
		#else
				class iterator_base {
		#endif
					public:
						typedef T                               value_type;
						typedef T*                              pointer;
						typedef T&                              reference;
						typedef size_t                          size_type;
						typedef ptrdiff_t                       difference_type;
						typedef std::bidirectional_iterator_tag iterator_category;

						iterator_base();
						iterator_base(tree_node *);

						T&             operator*() const;
						T*             operator->() const;

		            /// When called, the next increment/decrement skips children of this node.
						void         skip_children();
						void         skip_children(bool skip);
						/// Number of children of the node pointed to by the iterator.
						unsigned int number_of_children() const;

						sibling_iterator begin() const;
						sibling_iterator end() const;

						tree_node *node;
					protected:
						bool skip_current_children_;
				};

				/// Depth-first iterator, first accessing the node, then its children.
				class pre_order_iterator : public iterator_base { 
					public:
						pre_order_iterator();
						pre_order_iterator(tree_node *);
						pre_order_iterator(const iterator_base&);
						pre_order_iterator(const sibling_iterator&);

						bool    operator==(const pre_order_iterator&) const;
						bool    operator!=(const pre_order_iterator&) const;
						pre_order_iterator&  operator++();
					   	pre_order_iterator&  operator--();
						pre_order_iterator   operator++(int);
						pre_order_iterator   operator--(int);
						pre_order_iterator&  operator+=(unsigned int);
						pre_order_iterator&  operator-=(unsigned int);

						pre_order_iterator&  next_skip_children();
				};

				/*由于是普通树结构，不存在中序，子节点数目可大于2*/

				/// Depth-first iterator, first accessing the children, then the node itself.
				class post_order_iterator : public iterator_base {
					public:
						post_order_iterator();
						post_order_iterator(tree_node *);
						post_order_iterator(const iterator_base&);
						post_order_iterator(const sibling_iterator&);

						bool    operator==(const post_order_iterator&) const;
						bool    operator!=(const post_order_iterator&) const;
						post_order_iterator&  operator++();
					   	post_order_iterator&  operator--();
						post_order_iterator   operator++(int);
						post_order_iterator   operator--(int);
						post_order_iterator&  operator+=(unsigned int);
						post_order_iterator&  operator-=(unsigned int);

						/// Set iterator to the first child as deep as possible down the tree.
						void descend_all();
				};

				/// Breadth-first iterator, using a queue
				class breadth_first_queued_iterator : public iterator_base {
					public:
						breadth_first_queued_iterator();
						breadth_first_queued_iterator(tree_node *);
						breadth_first_queued_iterator(const iterator_base&);

						bool    operator==(const breadth_first_queued_iterator&) const;
						bool    operator!=(const breadth_first_queued_iterator&) const;
						breadth_first_queued_iterator&  operator++();
						breadth_first_queued_iterator   operator++(int);
						breadth_first_queued_iterator&  operator+=(unsigned int);

					private:
						std::queue<tree_node *> traversal_queue;
				};

				/// The default iterator types throughout the tree class.
				typedef pre_order_iterator            iterator;
				typedef breadth_first_queued_iterator breadth_first_iterator;

				/// Iterator which traverses only the nodes at a given depth from the root.
				class fixed_depth_iterator : public iterator_base {
					public:
						fixed_depth_iterator();
						fixed_depth_iterator(tree_node *);
						fixed_depth_iterator(const iterator_base&);
						fixed_depth_iterator(const sibling_iterator&);
						fixed_depth_iterator(const fixed_depth_iterator&);

						bool    operator==(const fixed_depth_iterator&) const;
						bool    operator!=(const fixed_depth_iterator&) const;
						fixed_depth_iterator&  operator++();
					   fixed_depth_iterator&  operator--();
						fixed_depth_iterator   operator++(int);
						fixed_depth_iterator   operator--(int);
						fixed_depth_iterator&  operator+=(unsigned int);
						fixed_depth_iterator&  operator-=(unsigned int);

						tree_node *top_node;
				};

				/// Iterator which traverses only the nodes which are siblings of each other.
				class sibling_iterator : public iterator_base {
					public:
						sibling_iterator();
						sibling_iterator(tree_node *);
						sibling_iterator(const sibling_iterator&);
						sibling_iterator(const iterator_base&);

						bool    operator==(const sibling_iterator&) const;
						bool    operator!=(const sibling_iterator&) const;
						sibling_iterator&  operator++();
						sibling_iterator&  operator--();
						sibling_iterator   operator++(int);
						sibling_iterator   operator--(int);
						sibling_iterator&  operator+=(unsigned int);
						sibling_iterator&  operator-=(unsigned int);

						tree_node *range_first() const;
						tree_node *range_last() const;
						tree_node *parent_;
					private:
						void set_parent_();
				};

		      	/// Iterator which traverses only the leaves.
		      	class leaf_iterator : public iterator_base {
		      	   public:
		      	      leaf_iterator();
		      	      leaf_iterator(tree_node *, tree_node *top=0);
		      	      leaf_iterator(const sibling_iterator&);
		      	      leaf_iterator(const iterator_base&);

		      	      bool    operator==(const leaf_iterator&) const;
		      	      bool    operator!=(const leaf_iterator&) const;
		      	      leaf_iterator&  operator++();
		      	      leaf_iterator&  operator--();
		      	      leaf_iterator   operator++(int);
		      	      leaf_iterator   operator--(int);
		      	      leaf_iterator&  operator+=(unsigned int);
		      	      leaf_iterator&  operator-=(unsigned int);
					private:
						tree_node *top_node;
		      	};

				/// Return iterator to the beginning of the tree.
				inline pre_order_iterator begin() const;
				/// Return iterator to the end of the tree.
				inline pre_order_iterator end() const;
				/// Return post-order iterator to the beginning of the tree.
				post_order_iterator begin_post() const;
				/// Return post-order end iterator of the tree.
				post_order_iterator end_post() const;
				/// Return fixed-depth iterator to the first node at a given depth from the given iterator.
				/// If 'walk_back=true', a depth=0 iterator will be taken from the beginning of the sibling
				/// range, not the current node.
				fixed_depth_iterator begin_fixed(const iterator_base&, unsigned int, bool walk_back=true) const;
				/// Return fixed-depth end iterator.
				fixed_depth_iterator end_fixed(const iterator_base&, unsigned int) const;
				/// Return breadth-first iterator to the first node at a given depth.
				breadth_first_queued_iterator begin_breadth_first() const;
				/// Return breadth-first end iterator.
				breadth_first_queued_iterator end_breadth_first() const;
				/// Return sibling iterator to the first child of given node.
				static sibling_iterator begin(const iterator_base&);
				/// Return sibling end iterator for children of given node.
				static sibling_iterator end(const iterator_base&);
		      	/// Return leaf iterator to the first leaf of the tree.
		      	leaf_iterator begin_leaf() const;
		      	/// Return leaf end iterator for entire tree.
		      	leaf_iterator end_leaf() const;
		      	/// Return leaf iterator to the first leaf of the subtree at the given node.
		      	leaf_iterator begin_leaf(const iterator_base& top) const;
		      	/// Return leaf end iterator for the subtree at the given node.
		      	leaf_iterator end_leaf(const iterator_base& top) const;

				typedef std::vector<int> path_t;
				/// Return a path (to be taken from the 'top' node) corresponding to a node in the tree.
				/// The first integer in path_t is the number of steps you need to go 'right' in the sibling
				/// chain (so 0 if we go straight to the children).
				path_t path_from_iterator(const iterator_base& iter, const iterator_base& top) const;
				/// Return an iterator given a path from the 'top' node.
				iterator iterator_from_path(const path_t&, const iterator_base& top) const;

				/// Return iterator to the parent of a node. Throws a `navigation_error` if the node
				/// does not have a parent.
				template<typename	iter> static iter parent(iter);
				/// Return iterator to the previous sibling of a node.
				template<typename iter> static iter previous_sibling(iter);
				/**
				 * \brief 获取兄弟节点，静态方法，因为关系结构已经固化在tree_node中，而iter就包含tree_node
				 * \note 
				 * 	Return iterator to the next sibling of a node.
				 * \author none
				 * \param[in] iter 目标
				 * \return 返回兄弟节点
				 * \retval retval
				 * \since version
				 * */
				template<typename iter> static iter next_sibling(iter);
				/// Return iterator to the next node at a given depth.
				template<typename iter> iter next_at_same_depth(iter) const;

				/// Erase all nodes of the tree.
				void clear();
				/// Erase element at position pointed to by iterator, return incremented iterator.
				template<typename iter> iter erase(iter);
				/// Erase all children of the node pointed to by iterator.
				void erase_children(const iterator_base&);
				/// Erase all siblings to the right of the iterator.
				void erase_right_siblings(const iterator_base&);
				/// Erase all siblings to the left of the iterator.
				void erase_left_siblings(const iterator_base&);

				/// Insert empty node as last/first child of node pointed to by position.
				template<typename iter> iter append_child(iter position); 
				template<typename iter> iter prepend_child(iter position); 
				/// Insert node as last/first child of node pointed to by position.
				template<typename iter> iter append_child(iter position, const T& x);
				template<typename iter> iter append_child(iter position, T&& x);
				template<typename iter> iter prepend_child(iter position, const T& x);
				template<typename iter> iter prepend_child(iter position, T&& x);
				/// Append the node (plus its children) at other_position as last/first child of position.
				template<typename iter> iter append_child(iter position, iter other_position);
				template<typename iter> iter prepend_child(iter position, iter other_position);
				/// Append the nodes in the from-to range (plus their children) as last/first children of position.
				template<typename iter> iter append_children(iter position, sibling_iterator from, sibling_iterator to);
				template<typename iter> iter prepend_children(iter position, sibling_iterator from, sibling_iterator to);

				/// Short-hand to insert topmost node in otherwise empty tree.
				pre_order_iterator set_head(const T& x);
				pre_order_iterator set_head(T&& x);
				/// Insert node as previous sibling of node pointed to by position.
				template<typename iter> iter insert(iter position, const T& x);
				template<typename iter> iter insert(iter position, T&& x);
				/// Specialisation of previous member.
				sibling_iterator insert(sibling_iterator position, const T& x);
				/// Insert node (with children) pointed to by subtree as previous sibling of node pointed to by position.
				/// Does not change the subtree itself (use move_in or move_in_below for that).
				template<typename iter> iter insert_subtree(iter position, const iterator_base& subtree);
				/// Insert node as next sibling of node pointed to by position.
				template<typename iter> iter insert_after(iter position, const T& x);
				template<typename iter> iter insert_after(iter position, T&& x);
				/// Insert node (with children) pointed to by subtree as next sibling of node pointed to by position.
				template<typename iter> iter insert_subtree_after(iter position, const iterator_base& subtree);

				/// Replace node at 'position' with other node (keeping same children); 'position' becomes invalid.
				template<typename iter> iter replace(iter position, const T& x);
				/// Replace node at 'position' with subtree starting at 'from' (do not erase subtree at 'from'); see above.
				template<typename iter> iter replace(iter position, const iterator_base& from);
				/// Replace string of siblings (plus their children) with copy of a new string (with children); see above
				sibling_iterator replace(sibling_iterator orig_begin, sibling_iterator orig_end, 
												 sibling_iterator new_begin,  sibling_iterator new_end); 

				/// Move all children of node at 'position' to be siblings, returns position.
				template<typename iter> iter flatten(iter position);
				/// Move nodes in range to be children of 'position'.
				template<typename iter> iter reparent(iter position, sibling_iterator begin, sibling_iterator end);
				/// Move all child nodes of 'from' to be children of 'position'.
				template<typename iter> iter reparent(iter position, iter from);

				/// Replace node with a new node, making the old node (plus subtree) a child of the new node.
				template<typename iter> iter wrap(iter position, const T& x);
				/// Replace the range of sibling nodes (plus subtrees), making these children of the new node.
				template<typename iter> iter wrap(iter from, iter to, const T& x);

				/// Move 'source' node (plus its children) to become the next sibling of 'target'.
				template<typename iter> iter move_after(iter target, iter source);
				/// Move 'source' node (plus its children) to become the previous sibling of 'target'.
		      	template<typename iter> iter move_before(iter target, iter source);
		      	sibling_iterator move_before(sibling_iterator target, sibling_iterator source);
				/// Move 'source' node (plus its children) to become the node at 'target' (erasing the node at 'target').
				template<typename iter> iter move_ontop(iter target, iter source);

				/// Extract the subtree starting at the indicated node, removing it from the original tree.
				tree                         move_out(iterator);
				/// Inverse of take_out: inserts the given tree as previous sibling of indicated node by a 
				/// move operation, that is, the given tree becomes empty. Returns iterator to the top node.
				template<typename iter> iter move_in(iter, tree&);
				/// As above, but now make the tree the last child of the indicated node.
				template<typename iter> iter move_in_below(iter, tree&);
				/// As above, but now make the tree the nth child of the indicated node (if possible).
				template<typename iter> iter move_in_as_nth_child(iter, size_t, tree&);

				/// Merge with other tree, creating new branches and leaves only if they are not already present.
				void merge(sibling_iterator, sibling_iterator, sibling_iterator, sibling_iterator, 
									bool duplicate_leaves=false);
				/// As above, but using two trees with a single top node at the 'to' and 'from' positions.
				void merge(iterator to, iterator from, bool duplicate_leaves);
				/// Sort (std::sort only moves values of nodes, this one moves children as well).
				void sort(sibling_iterator from, sibling_iterator to, bool deep=false);
				template<class StrictWeakOrdering>
				void sort(sibling_iterator from, sibling_iterator to, StrictWeakOrdering comp, bool deep=false);
				/// Compare two ranges of nodes (compares nodes as well as tree structure).
				template<typename iter>
				bool equal(const iter& one, const iter& two, const iter& three) const;
				template<typename iter, class BinaryPredicate>
				bool equal(const iter& one, const iter& two, const iter& three, BinaryPredicate) const;
				template<typename iter>
				bool equal_subtree(const iter& one, const iter& two) const;
				template<typename iter, class BinaryPredicate>
				bool equal_subtree(const iter& one, const iter& two, BinaryPredicate) const;
				/// Extract a new tree formed by the range of siblings plus all their children.
				tree subtree(sibling_iterator from, sibling_iterator to) const;
				void subtree(tree&, sibling_iterator from, sibling_iterator to) const;
				/// Exchange the node (plus subtree) with its sibling node (do nothing if no sibling present).
				void swap(sibling_iterator it);
				/// Exchange two nodes (plus subtrees). The iterators will remain valid and keep 
				/// pointing to the same nodes, which now sit at different locations in the tree.
			   	void swap(iterator, iterator);

				/// Count the total number of nodes.
				size_t size() const;
				/// Count the total number of nodes below the indicated node (plus one).
				size_t size(const iterator_base&) const;
				/// Check if tree is empty.
				bool empty() const;
				/// Compute the depth to the root or to a fixed other iterator.(查询目标节点到指定某一节点的距离，根节点也可以是指定的节点)
				static int depth(const iterator_base&);
				static int depth(const iterator_base&, const iterator_base&);
				/**
				 * \brief depth 计算指定节点到root的步数
				 * \note Compute the depth to the root, counting all levels for which predicate returns true.
				 * \author none
				 * \param[in] p 是某一些断言，可以接收iterator_base对象，并作出false与true的断言，如果是true，则返回值要自增一
				 * \param[in] iterator_base 是一个迭代器，对应节点位置
				 * \return return
				 * \retval retval
				 * \since v0.0.1
				 * */
				template<class Predicate>
				static int depth(const iterator_base&, Predicate p);
				/// Compute the depth distance between two nodes, counting all levels for which predicate returns true.
				template<class Predicate>
				static int distance(const iterator_base& top, const iterator_base& bottom, Predicate p);
				/// Determine the maximal depth of the tree. An empty tree has max_depth=-1.
				int max_depth() const;
				/// Determine the maximal depth of the tree with top node at the given position.
				int max_depth(const iterator_base&) const;
				/// Count the number of children of node at position.
				static unsigned int number_of_children(const iterator_base&);
				/// Count the number of siblings (left and right) of node at iterator. Total nodes at this level is +1.
				unsigned int number_of_siblings(const iterator_base&) const;
				/// Determine whether node at position is in the subtrees with indicated top node.
		   		bool     is_in_subtree(const iterator_base& position, const iterator_base& top) const;
				/// Determine whether node at position is in the subtrees with root in the range.
				bool     is_in_subtree(const iterator_base& position, const iterator_base& begin, 
											  const iterator_base& end) const;
				/// Determine whether the iterator is an 'end' iterator and thus not actually pointing to a node.
				bool     is_valid(const iterator_base&) const;
				/// Determine whether the iterator is one of the 'head' nodes at the top level, i.e. has no parent.
				static   bool is_head(const iterator_base&);
				/// Find the lowest common ancestor of two nodes, that is, the deepest node such that
				/// both nodes are descendants of it.
				iterator lowest_common_ancestor(const iterator_base&, const iterator_base &) const;

				/// Determine the index of a node in the range of siblings to which it belongs.
				unsigned int index(sibling_iterator it) const;
				/// Inverse of 'index': return the n-th child of the node at position.
				static sibling_iterator child(const iterator_base& position, unsigned int);
				/// Return iterator to the sibling indicated by index
				sibling_iterator sibling(const iterator_base& position, unsigned int) const;  				

				/// For debugging only: verify internal consistency by inspecting all pointers in the tree
				/// (which will also trigger a valgrind error in case something got corrupted).
				void debug_verify_consistency() const;

				/// Comparator class for iterators (compares pointer values; why doesn't this work automatically?)
				class iterator_base_less {
					public:
						bool operator()(const typename tree<T, tree_nodeallocator>::iterator_base& one,
											 const typename tree<T, tree_nodeallocator>::iterator_base& two) const
							{
							return one.node < two.node;
							}
				};
				tree_node *head, *feet;    // head/feet are always dummy; if an iterator points to them it is invalid
			private:
				tree_nodeallocator alloc_;
				void head_initialise_();
				void copy_(const tree<T, tree_nodeallocator>& other);

		      	/// Comparator class for two nodes of a tree (used for sorting and searching).
				template<class StrictWeakOrdering>
				class compare_nodes {
					public:
						compare_nodes(StrictWeakOrdering comp) : comp_(comp) {};

						bool operator()(const tree_node *a, const tree_node *b) 
							{
							return comp_(a->data, b->data);
							}
					private:
						StrictWeakOrdering comp_;
				};
			};

		//template <class T, class tree_nodeallocator>
		//class iterator_base_less {
		//	public:
		//		bool operator()(const typename tree<T, tree_nodeallocator>::iterator_base& one,
		//						  const typename tree<T, tree_nodeallocator>::iterator_base& two) const
		//			{
		//			txtout << "operatorclass<" << one.node < two.node << std::endl;
		//			return one.node < two.node;
		//			}
		//};

		// template <class T, class tree_nodeallocator>
		// bool operator<(const typename tree<T, tree_nodeallocator>::iterator& one,
		// 					const typename tree<T, tree_nodeallocator>::iterator& two)
		// 	{
		// 	txtout << "operator< " << one.node < two.node << std::endl;
		// 	if(one.node < two.node) return true;
		// 	return false;
		// 	}
		// 
		// template <class T, class tree_nodeallocator>
		// bool operator==(const typename tree<T, tree_nodeallocator>::iterator& one,
		// 					const typename tree<T, tree_nodeallocator>::iterator& two)
		// 	{
		// 	txtout << "operator== " << one.node == two.node << std::endl;
		// 	if(one.node == two.node) return true;
		// 	return false;
		// 	}
		// 
		// template <class T, class tree_nodeallocator>
		// bool operator>(const typename tree<T, tree_nodeallocator>::iterator_base& one,
		// 					const typename tree<T, tree_nodeallocator>::iterator_base& two)
		// 	{
		// 	txtout << "operator> " << one.node < two.node << std::endl;
		// 	if(one.node > two.node) return true;
		// 	return false;
		// 	}



		// Tree

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::tree() 
			{
			head_initialise_();
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::tree(const T& x) 
			{
			head_initialise_();
			set_head(x);
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::tree(tree<T, tree_nodeallocator>&& x) {
			head_initialise_();
			if(x.head->_next_sibling != x.feet) { // move tree if non-empty only
				head->_next_sibling =x.head->_next_sibling;
				feet->_prev_sibling = x.feet->_prev_sibling;
				x.head->_next_sibling->_prev_sibling = head;
				x.feet->_prev_sibling->_next_sibling = feet;
				x.head->_next_sibling = x.feet;
				x.feet->_prev_sibling = x.head;
			}
		}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::tree(const iterator_base& other) {
			head_initialise_();
			set_head((*other));
			replace(begin(), other);
		}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::~tree()
			{
			clear();
			std::allocator_traits<decltype(alloc_)>::destroy(alloc_, head);
			std::allocator_traits<decltype(alloc_)>::destroy(alloc_, feet);
			std::allocator_traits<decltype(alloc_)>::deallocate(alloc_, head, 1);
			std::allocator_traits<decltype(alloc_)>::deallocate(alloc_, feet, 1);
			}

		template <class T, class tree_nodeallocator>
		void tree<T, tree_nodeallocator>::head_initialise_() {
			head = std::allocator_traits<decltype(alloc_)>::allocate(alloc_, 1, 0);
		   	feet = std::allocator_traits<decltype(alloc_)>::allocate(alloc_, 1, 0);
			std::allocator_traits<decltype(alloc_)>::construct(alloc_, head, tree_node());
			std::allocator_traits<decltype(alloc_)>::construct(alloc_, feet, tree_node());	

		   	head->_parent=0;
		   	head->_first_child=0;
		   	head->_last_child=0;
		   	head->_prev_sibling=0; //head;
		   	head->_next_sibling=feet; //head;

			feet->_parent=0;
			feet->_first_child=0;
			feet->_last_child=0;
			feet->_prev_sibling=head;
			feet->_next_sibling=0;
		}

		template <class T, class tree_nodeallocator>
		tree<T,tree_nodeallocator>& tree<T, tree_nodeallocator>::operator=(const tree<T, tree_nodeallocator>& other)
			{
			if(this != &other)
				copy_(other);
			return *this;
			}

		template <class T, class tree_nodeallocator>
		tree<T,tree_nodeallocator>& tree<T, tree_nodeallocator>::operator=(tree<T, tree_nodeallocator>&& x)
			{
			if(this != &x) {
				clear(); // clear any existing data.

				head->_next_sibling=x.head->_next_sibling;
				feet->_prev_sibling=x.feet->_prev_sibling;
				x.head->_next_sibling->_prev_sibling=head;
				x.feet->_prev_sibling->_next_sibling=feet;
				x.head->_next_sibling=x.feet;
				x.feet->_prev_sibling=x.head;
				}
			return *this;
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::tree(const tree<T, tree_nodeallocator>& other)
			{
			head_initialise_();
			copy_(other);
			}

		template <class T, class tree_nodeallocator>
		void tree<T, tree_nodeallocator>::copy_(const tree<T, tree_nodeallocator>& other) 
			{
			clear();
			pre_order_iterator it=other.begin(), to=begin();
			while(it!=other.end()) {
				to=insert(to, (*it));
				it.skip_children();
				++it;
				}
			to=begin();
			it=other.begin();
			while(it!=other.end()) {
				to=replace(to, it);
				to.skip_children();
				it.skip_children();
				++to;
				++it;
				}
			}

		template <class T, class tree_nodeallocator>
		void tree<T, tree_nodeallocator>::clear()
			{
			if(head)
				while(head->_next_sibling!=feet)
					erase(pre_order_iterator(head->_next_sibling));
			}

		template<class T, class tree_nodeallocator> 
		void tree<T, tree_nodeallocator>::erase_children(const iterator_base& it)
			{
		//	std::cout << "erase_children " << it.node << std::endl;
			if(it.node==0) return;

			tree_node *cur=it.node->_first_child;
			tree_node *prev=0;

			while(cur!=0) {
				prev=cur;
				cur=cur->_next_sibling;
				erase_children(pre_order_iterator(prev));
				std::allocator_traits<decltype(alloc_)>::destroy(alloc_, prev);
				std::allocator_traits<decltype(alloc_)>::deallocate(alloc_, prev, 1);
				}
			it.node->_first_child=0;
			it.node->_last_child=0;
		//	std::cout << "exit" << std::endl;
			}

		template<class T, class tree_nodeallocator> 
		void tree<T, tree_nodeallocator>::erase_right_siblings(const iterator_base& it)
			{
			if(it.node==0) return;

			tree_node *cur=it.node->_next_sibling;
			tree_node *prev=0;

			while(cur!=0) {
				prev=cur;
				cur=cur->_next_sibling;
				erase_children(pre_order_iterator(prev));
				std::allocator_traits<decltype(alloc_)>::destroy(alloc_, prev);
				std::allocator_traits<decltype(alloc_)>::deallocate(alloc_, prev, 1);
				}
			it.node->_next_sibling=0;
			if(it.node->_parent!=0)
				it.node->_parent->_last_child=it.node;
			}

		template<class T, class tree_nodeallocator> 
		void tree<T, tree_nodeallocator>::erase_left_siblings(const iterator_base& it)
			{
			if(it.node==0) return;

			tree_node *cur=it.node->_prev_sibling;
			tree_node *prev=0;

			while(cur!=0) {
				prev=cur;
				cur=cur->_prev_sibling;
				erase_children(pre_order_iterator(prev));
				std::allocator_traits<decltype(alloc_)>::destroy(alloc_, prev);
				std::allocator_traits<decltype(alloc_)>::deallocate(alloc_, prev, 1);
				}
			it.node->_prev_sibling=0;
			if(it.node->_parent!=0)
				it.node->_parent->_first_child=it.node;
			}

		template<class T, class tree_nodeallocator> 
		template<class iter>
		iter tree<T, tree_nodeallocator>::erase(iter it)
			{
			tree_node *cur=it.node;
			assert(cur!=head);
			iter ret=it;
			ret.skip_children();
			++ret;
			erase_children(it);
			if(cur->_prev_sibling==0) {
				cur->_parent->_first_child=cur->_next_sibling;
				}
			else {
				cur->_prev_sibling->_next_sibling=cur->_next_sibling;
				}
			if(cur->_next_sibling==0) {
				cur->_parent->_last_child=cur->_prev_sibling;
				}
			else {
				cur->_next_sibling->_prev_sibling=cur->_prev_sibling;
				}

			std::allocator_traits<decltype(alloc_)>::destroy(alloc_, cur);
			std::allocator_traits<decltype(alloc_)>::deallocate(alloc_, cur, 1);
			return ret;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::pre_order_iterator tree<T, tree_nodeallocator>::begin() const
			{
			return pre_order_iterator(head->_next_sibling);
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::pre_order_iterator tree<T, tree_nodeallocator>::end() const
			{
			return pre_order_iterator(feet);
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::breadth_first_queued_iterator tree<T, tree_nodeallocator>::begin_breadth_first() const
			{
			return breadth_first_queued_iterator(head->_next_sibling);
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::breadth_first_queued_iterator tree<T, tree_nodeallocator>::end_breadth_first() const
			{
			return breadth_first_queued_iterator();
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::post_order_iterator tree<T, tree_nodeallocator>::begin_post() const
			{
			tree_node *tmp=head->_next_sibling;
			if(tmp!=feet) {
				while(tmp->_first_child)
					tmp=tmp->_first_child;
				}
			return post_order_iterator(tmp);
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::post_order_iterator tree<T, tree_nodeallocator>::end_post() const
			{
			return post_order_iterator(feet);
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::fixed_depth_iterator tree<T, tree_nodeallocator>::begin_fixed(const iterator_base& pos, unsigned int dp, bool walk_back) const
			{
			typename tree<T, tree_nodeallocator>::fixed_depth_iterator ret;
			ret.top_node=pos.node;

			tree_node *tmp=pos.node;
			unsigned int curdepth=0;
			while(curdepth<dp) { // go down one level
				while(tmp->_first_child==0) {
					if(tmp->_next_sibling==0) {
						// try to walk up and then right again
						do {
							if(tmp==ret.top_node)
							   throw std::range_error("tree: begin_fixed out of range");
							tmp=tmp->_parent;
		               if(tmp==0) 
							   throw std::range_error("tree: begin_fixed out of range");
		               --curdepth;
						   } while(tmp->_next_sibling==0);
						}
					tmp=tmp->_next_sibling;
					}
				tmp=tmp->_first_child;
				++curdepth;
				}

			// Now walk back to the first sibling in this range.
			if(walk_back)
			while(tmp->_prev_sibling!=0)
				tmp=tmp->_prev_sibling;	

			ret.node=tmp;
			return ret;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::fixed_depth_iterator tree<T, tree_nodeallocator>::end_fixed(const iterator_base& pos, unsigned int dp) const
			{
			assert(1==0); // FIXME: not correct yet: use is_valid() as a temporary workaround 
			tree_node *tmp=pos.node;
			unsigned int curdepth=1;
			while(curdepth<dp) { // go down one level
				while(tmp->_first_child==0) {
					tmp=tmp->_next_sibling;
					if(tmp==0)
						throw std::range_error("tree: end_fixed out of range");
					}
				tmp=tmp->_first_child;
				++curdepth;
				}
			return tmp;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::sibling_iterator tree<T, tree_nodeallocator>::begin(const iterator_base& pos) 
			{
			assert(pos.node!=0);
			if(pos.node->_first_child==0) {
				return end(pos);
				}
			return pos.node->_first_child;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::sibling_iterator tree<T, tree_nodeallocator>::end(const iterator_base& pos) 
			{
			sibling_iterator ret(0);
			ret.parent_=pos.node;
			return ret;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::leaf_iterator tree<T, tree_nodeallocator>::begin_leaf() const
		   {
		   tree_node *tmp=head->_next_sibling;
		   if(tmp!=feet) {
		      while(tmp->_first_child)
		         tmp=tmp->_first_child;
		      }
		   return leaf_iterator(tmp);
		   }

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::leaf_iterator tree<T, tree_nodeallocator>::end_leaf() const
		   {
		   return leaf_iterator(feet);
		   }

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::path_t tree<T, tree_nodeallocator>::path_from_iterator(const iterator_base& iter, const iterator_base& top) const
			{
			path_t path;
			tree_node *walk=iter.node;

			do {
				if(path.size()>0)
					walk=walk->_parent;
				int num=0;
				while(walk!=top.node && walk->_prev_sibling!=0 && walk->_prev_sibling!=head) {
					++num;
					walk=walk->_prev_sibling;
					}
				path.push_back(num);
				}
			while(walk->_parent!=0 && walk!=top.node);

			std::reverse(path.begin(), path.end());
			return path;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::iterator tree<T, tree_nodeallocator>::iterator_from_path(const path_t& path, const iterator_base& top) const
			{
			iterator it=top;
			tree_node *walk=it.node;

			for(size_t step=0; step<path.size(); ++step) {
				if(step>0)
					walk=walk->_first_child;
				if(walk==0)
					throw std::range_error("tree::iterator_from_path: no more nodes at step "+std::to_string(step));

				for(int i=0; i<path[step]; ++i) {
					walk=walk->_next_sibling;
					if(walk==0)
						throw std::range_error("tree::iterator_from_path: out of siblings at step "+std::to_string(step));
					}
				}
			it.node=walk;
			return it;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::leaf_iterator tree<T, tree_nodeallocator>::begin_leaf(const iterator_base& top) const
		   {
			tree_node *tmp=top.node;
			while(tmp->_first_child)
				 tmp=tmp->_first_child;
		   return leaf_iterator(tmp, top.node);
		   }

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::leaf_iterator tree<T, tree_nodeallocator>::end_leaf(const iterator_base& top) const
		   {
		   return leaf_iterator(top.node, top.node);
		   }

		template <class T, class tree_nodeallocator>
		template <typename iter>
		iter tree<T, tree_nodeallocator>::parent(iter position) 
			{
			if(position.node==0)
				throw navigation_error("tree: attempt to navigate from null iterator.");

			if(position.node->_parent==0) 
				throw navigation_error("tree: attempt to navigate up past head node.");

			return iter(position.node->_parent);
			}

		template <class T, class tree_nodeallocator>
		template <typename iter>
		iter tree<T, tree_nodeallocator>::previous_sibling(iter position) 
			{
			assert(position.node!=0);
			iter ret(position);
			ret.node=position.node->_prev_sibling;
			return ret;
			}

		template <class T, class tree_nodeallocator>
		template <typename iter>
		iter tree<T, tree_nodeallocator>::next_sibling(iter position) 
			{
			assert(position.node!=0);
			iter ret(position);
			ret.node=position.node->_next_sibling;
			return ret;
			}

		template <class T, class tree_nodeallocator>
		template <typename iter>
		iter tree<T, tree_nodeallocator>::next_at_same_depth(iter position) const
			{
			// We make use of a temporary fixed_depth iterator to implement this.

			typename tree<T, tree_nodeallocator>::fixed_depth_iterator tmp(position.node);

			++tmp;
			return iter(tmp);

			//	assert(position.node!=0);
			//	iter ret(position);
			//
			//	if(position.node->_next_sibling) {
			//		ret.node=position.node->_next_sibling;
			//		}
			//	else {
			//		int relative_depth=0;
			//	   upper:
			//		do {
			//			ret.node=ret.node->_parent;
			//			if(ret.node==0) return ret;
			//			--relative_depth;
			//			} while(ret.node->_next_sibling==0);
			//	   lower:
			//		ret.node=ret.node->_next_sibling;
			//		while(ret.node->_first_child==0) {
			//			if(ret.node->_next_sibling==0)
			//				goto upper;
			//			ret.node=ret.node->_next_sibling;
			//			if(ret.node==0) return ret;
			//			}
			//		while(relative_depth<0 && ret.node->_first_child!=0) {
			//			ret.node=ret.node->_first_child;
			//			++relative_depth;
			//			}
			//		if(relative_depth<0) {
			//			if(ret.node->_next_sibling==0) goto upper;
			//			else                          goto lower;
			//			}
			//		}
			//	return ret;
			}

		/**
		 * \brief append_child 在子节点集合末端增加一个子节点并将该子节点的引用iter返回
		 * \note 
		 * \author none
		 * \param[in] position 目标父节点
		 * \return iter 返回增加的子节点额引用
		 * \since v0.0.1
		 * */
		template <class T, class tree_nodeallocator>
		template <typename iter>
		iter tree<T, tree_nodeallocator>::append_child(iter position)
		 	{
			assert(position.node!=head);
			assert(position.node!=feet);
			assert(position.node);

			tree_node *tmp=std::allocator_traits<decltype(alloc_)>::allocate(alloc_, 1, 0);
			std::allocator_traits<decltype(alloc_)>::construct(alloc_, tmp, tree_node());
			tmp->_first_child=0;
			tmp->_last_child=0;

			tmp->_parent=position.node;
			if(position.node->_last_child!=0) {
				position.node->_last_child->_next_sibling=tmp;
				}
			else {
				position.node->_first_child=tmp;
				}
			tmp->_prev_sibling=position.node->_last_child;
			position.node->_last_child=tmp;
			tmp->_next_sibling=0;
			return tmp;
		 	}

		/**
		 * \brief prepend_child 与append_child功能类似，增加子节点的位置变成了子节点集合的前端
		 * \note note
		 * \author none
		 * \param[in] position 目标父节点
		 * \return iter 返回增加的子节点的迭代器引用
		 * \retval retval
		 * \since v0.0.1
		 * */
		template <class T, class tree_nodeallocator>
		template <typename iter>
		iter tree<T, tree_nodeallocator>::prepend_child(iter position)
		 	{
			assert(position.node!=head);
			assert(position.node!=feet);
			assert(position.node);

			tree_node *tmp=std::allocator_traits<decltype(alloc_)>::allocate(alloc_, 1, 0);
			std::allocator_traits<decltype(alloc_)>::construct(alloc_, tmp, tree_node());
			tmp->_first_child=0;
			tmp->_last_child=0;

			tmp->_parent=position.node;
			if(position.node->_first_child!=0) {
				position.node->_first_child->_prev_sibling=tmp;
				}
			else {
				position.node->_last_child=tmp;
				}
			tmp->_next_sibling=position.node->_first_child;
			position.node->prev_child=tmp;
			tmp->_prev_sibling=0;
			return tmp;
		 	}

		/**
		 * \brief append_child 参考append_child，本函数直接传入节点的值，不需要外部使用引用迭代器去赋值
		 * \note note
		 * \author none
		 * \param[in] position 目标父节点
		 * \param[in] x 增加的子节点的值
		 * \return iter 增加子节点的迭代器引用
		 * \retval retval
		 * \since v0.0.1
		 * */
		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::append_child(iter position, const T& x)
			{
			// If your program fails here you probably used 'append_child' to add the top
			// node to an empty tree. From version 1.45 the top element should be added
			// using 'insert'. See the documentation for further information, and sorry about
			// the API change.
			assert(position.node!=head);
			assert(position.node!=feet);
			assert(position.node);

			tree_node *tmp=std::allocator_traits<decltype(alloc_)>::allocate(alloc_, 1, 0);
			std::allocator_traits<decltype(alloc_)>::construct(alloc_, tmp, x);
			tmp->_first_child=0;
			tmp->_last_child=0;

			tmp->_parent=position.node;
			if(position.node->_last_child!=0) {
				position.node->_last_child->_next_sibling=tmp;
				}
			else {
				position.node->_first_child=tmp;
				}
			tmp->_prev_sibling=position.node->_last_child;
			position.node->_last_child=tmp;
			tmp->_next_sibling=0;
			return tmp;
			}

		/**
		 * \brief append_child 参考
		 * \note note
		 * \author none
		 * \param[in] in
		 * \param[out] out
		 * \return return
		 * \retval retval
		 * \since version
		 * */
		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::append_child(iter position, T&& x)
			{
			assert(position.node!=head);
			assert(position.node!=feet);
			assert(position.node);

			tree_node *tmp=std::allocator_traits<decltype(alloc_)>::allocate(alloc_, 1, 0);
			std::allocator_traits<decltype(alloc_)>::construct(alloc_, tmp); // Here is where the move semantics kick in
			std::swap(tmp->data, x);

			tmp->_first_child=0;
			tmp->_last_child=0;

			tmp->_parent=position.node;
			if(position.node->_last_child!=0) {
				position.node->_last_child->_next_sibling=tmp;
				}
			else {
				position.node->_first_child=tmp;
				}
			tmp->_prev_sibling=position.node->_last_child;
			position.node->_last_child=tmp;
			tmp->_next_sibling=0;
			return tmp;
			}

		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::prepend_child(iter position, const T& x)
			{
			assert(position.node!=head);
			assert(position.node!=feet);
			assert(position.node);

			tree_node *tmp=std::allocator_traits<decltype(alloc_)>::allocate(alloc_, 1, 0);
			std::allocator_traits<decltype(alloc_)>::construct(alloc_, tmp, x);
			tmp->_first_child=0;
			tmp->_last_child=0;

			tmp->_parent=position.node;
			if(position.node->_first_child!=0) {
				position.node->_first_child->_prev_sibling=tmp;
				}
			else {
				position.node->_last_child=tmp;
				}
			tmp->_next_sibling=position.node->_first_child;
			position.node->_first_child=tmp;
			tmp->_prev_sibling=0;
			return tmp;
			}

		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::prepend_child(iter position, T&& x)
			{
			assert(position.node!=head);
			assert(position.node!=feet);
			assert(position.node);

			tree_node *tmp=std::allocator_traits<decltype(alloc_)>::allocate(alloc_, 1, 0);
			std::allocator_traits<decltype(alloc_)>::construct(alloc_, tmp);
			std::swap(tmp->data, x);

			tmp->_first_child=0;
			tmp->_last_child=0;

			tmp->_parent=position.node;
			if(position.node->_first_child!=0) {
				position.node->_first_child->_prev_sibling=tmp;
				}
			else {
				position.node->_last_child=tmp;
				}
			tmp->_next_sibling=position.node->_first_child;
			position.node->_first_child=tmp;
			tmp->_prev_sibling=0;
			return tmp;
			}

		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::append_child(iter position, iter other)
			{
			assert(position.node!=head);
			assert(position.node!=feet);
			assert(position.node);

			sibling_iterator aargh=append_child(position, value_type());
			return replace(aargh, other);
			}

		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::prepend_child(iter position, iter other)
			{
			assert(position.node!=head);
			assert(position.node!=feet);
			assert(position.node);

			sibling_iterator aargh=prepend_child(position, value_type());
			return replace(aargh, other);
			}

		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::append_children(iter position, sibling_iterator from, sibling_iterator to)
			{
			assert(position.node!=head);
			assert(position.node!=feet);
			assert(position.node);

			iter ret=from;

			while(from!=to) {
				insert_subtree(position.end(), from);
				++from;
				}
			return ret;
			}

		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::prepend_children(iter position, sibling_iterator from, sibling_iterator to)
			{
			assert(position.node!=head);
			assert(position.node!=feet);
			assert(position.node);

			if(from==to) return from; // should return end of tree?

			iter ret;
			do {
				--to;
				ret=insert_subtree(position.begin(), to);
				}
			while(to!=from);

			return ret;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::pre_order_iterator tree<T, tree_nodeallocator>::set_head(const T& x)
			{
			assert(head->_next_sibling==feet);
			return insert(iterator(feet), x);
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::pre_order_iterator tree<T, tree_nodeallocator>::set_head(T&& x)
			{
			assert(head->_next_sibling==feet);
			return insert(iterator(feet), x);
			}

		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::insert(iter position, const T& x)
			{
			if(position.node==0) {
				position.node=feet; // Backward compatibility: when calling insert on a null node,
				                    // insert before the feet.
				}
			assert(position.node!=head); // Cannot insert before head.

			tree_node *tmp=std::allocator_traits<decltype(alloc_)>::allocate(alloc_, 1, 0);
			std::allocator_traits<decltype(alloc_)>::construct(alloc_, tmp, x);
			tmp->_first_child=0;
			tmp->_last_child=0;

			tmp->_parent=position.node->_parent;
			tmp->_next_sibling=position.node;
			tmp->_prev_sibling=position.node->_prev_sibling;
			position.node->_prev_sibling=tmp;

			if(tmp->_prev_sibling==0) {
				if(tmp->_parent) // when inserting nodes at the head, there is no parent
					tmp->_parent->_first_child=tmp;
				}
			else
				tmp->_prev_sibling->_next_sibling=tmp;
			return tmp;
			}

		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::insert(iter position, T&& x)
			{
			if(position.node==0) {
				position.node=feet; // Backward compatibility: when calling insert on a null node,
				                    // insert before the feet.
				}
			tree_node *tmp=std::allocator_traits<decltype(alloc_)>::allocate(alloc_, 1, 0);
			std::allocator_traits<decltype(alloc_)>::construct(alloc_, tmp);
			std::swap(tmp->data, x); // Move semantics
			tmp->_first_child=0;
			tmp->_last_child=0;

			tmp->_parent=position.node->_parent;
			tmp->_next_sibling=position.node;
			tmp->_prev_sibling=position.node->_prev_sibling;
			position.node->_prev_sibling=tmp;

			if(tmp->_prev_sibling==0) {
				if(tmp->_parent) // when inserting nodes at the head, there is no parent
					tmp->_parent->_first_child=tmp;
				}
			else
				tmp->_prev_sibling->_next_sibling=tmp;
			return tmp;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::sibling_iterator tree<T, tree_nodeallocator>::insert(sibling_iterator position, const T& x)
			{
			tree_node *tmp=std::allocator_traits<decltype(alloc_)>::allocate(alloc_, 1, 0);
			std::allocator_traits<decltype(alloc_)>::construct(alloc_, tmp, x);
			tmp->_first_child=0;
			tmp->_last_child=0;

			tmp->_next_sibling=position.node;
			if(position.node==0) { // iterator points to end of a subtree
				tmp->_parent=position.parent_;
				tmp->_prev_sibling=position.range_last();
				tmp->_parent->_last_child=tmp;
				}
			else {
				tmp->_parent=position.node->_parent;
				tmp->_prev_sibling=position.node->_prev_sibling;
				position.node->_prev_sibling=tmp;
				}

			if(tmp->_prev_sibling==0) {
				if(tmp->_parent) // when inserting nodes at the head, there is no parent
					tmp->_parent->_first_child=tmp;
				}
			else
				tmp->_prev_sibling->_next_sibling=tmp;
			return tmp;
			}

		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::insert_after(iter position, const T& x)
			{
			tree_node *tmp=std::allocator_traits<decltype(alloc_)>::allocate(alloc_, 1, 0);
			std::allocator_traits<decltype(alloc_)>::construct(alloc_, tmp, x);
			tmp->_first_child=0;
			tmp->_last_child=0;

			tmp->_parent=position.node->_parent;
			tmp->_prev_sibling=position.node;
			tmp->_next_sibling=position.node->_next_sibling;
			position.node->_next_sibling=tmp;

			if(tmp->_next_sibling==0) {
				if(tmp->_parent) // when inserting nodes at the head, there is no parent
					tmp->_parent->_last_child=tmp;
				}
			else {
				tmp->_next_sibling->_prev_sibling=tmp;
				}
			return tmp;
			}

		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::insert_after(iter position, T&& x)
			{
			tree_node *tmp=std::allocator_traits<decltype(alloc_)>::allocate(alloc_, 1, 0);
			std::allocator_traits<decltype(alloc_)>::construct(alloc_, tmp);
			std::swap(tmp->data, x); // move semantics
			tmp->_first_child=0;
			tmp->_last_child=0;

			tmp->_parent=position.node->_parent;
			tmp->_prev_sibling=position.node;
			tmp->_next_sibling=position.node->_next_sibling;
			position.node->_next_sibling=tmp;

			if(tmp->_next_sibling==0) {
				if(tmp->_parent) // when inserting nodes at the head, there is no parent
					tmp->_parent->_last_child=tmp;
				}
			else {
				tmp->_next_sibling->_prev_sibling=tmp;
				}
			return tmp;
			}

		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::insert_subtree(iter position, const iterator_base& subtree)
			{
			// insert dummy
			iter it=insert(position, value_type());
			// replace dummy with subtree
			return replace(it, subtree);
			}

		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::insert_subtree_after(iter position, const iterator_base& subtree)
			{
			// insert dummy
			iter it=insert_after(position, value_type());
			// replace dummy with subtree
			return replace(it, subtree);
			}

		// template <class T, class tree_nodeallocator>
		// template <class iter>
		// iter tree<T, tree_nodeallocator>::insert_subtree(sibling_iterator position, iter subtree)
		// 	{
		// 	// insert dummy
		// 	iter it(insert(position, value_type()));
		// 	// replace dummy with subtree
		// 	return replace(it, subtree);
		// 	}

		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::replace(iter position, const T& x)
			{
		//	kp::destructor(&position.node->data);
		//	kp::constructor(&position.node->data, x);
			position.node->data=x;
		//	alloc_.destroy(position.node);
		//	alloc_.construct(position.node, x);
			return position;
			}

		template <class T, class tree_nodeallocator>
		template <class iter>
		iter tree<T, tree_nodeallocator>::replace(iter position, const iterator_base& from)
			{
			assert(position.node!=head);
			tree_node *current_from=from.node;
			tree_node *start_from=from.node;
			tree_node *current_to  =position.node;

			// replace the node at position with head of the replacement tree at from
		//	std::cout << "warning!" << position.node << std::endl;
			erase_children(position);	
		//	std::cout << "no warning!" << std::endl;
			tree_node *tmp=std::allocator_traits<decltype(alloc_)>::allocate(alloc_, 1, 0);
			std::allocator_traits<decltype(alloc_)>::construct(alloc_, tmp, (*from));
			tmp->_first_child=0;
			tmp->_last_child=0;
			if(current_to->_prev_sibling==0) {
				if(current_to->_parent!=0)
					current_to->_parent->_first_child=tmp;
				}
			else {
				current_to->_prev_sibling->_next_sibling=tmp;
				}
			tmp->_prev_sibling=current_to->_prev_sibling;
			if(current_to->_next_sibling==0) {
				if(current_to->_parent!=0)
					current_to->_parent->_last_child=tmp;
				}
			else {
				current_to->_next_sibling->_prev_sibling=tmp;
				}
			tmp->_next_sibling=current_to->_next_sibling;
			tmp->_parent=current_to->_parent;
		//	kp::destructor(&current_to->data);
			std::allocator_traits<decltype(alloc_)>::destroy(alloc_, current_to);
			std::allocator_traits<decltype(alloc_)>::deallocate(alloc_, current_to, 1);
			current_to=tmp;

			// only at this stage can we fix 'last'
			tree_node *last=from.node->_next_sibling;

			pre_order_iterator toit=tmp;
			// copy all children
			do {
				assert(current_from!=0);
				if(current_from->_first_child != 0) {
					current_from=current_from->_first_child;
					toit=append_child(toit, current_from->data);
					}
				else {
					while(current_from->_next_sibling==0 && current_from!=start_from) {
						current_from=current_from->_parent;
						toit=parent(toit);
						assert(current_from!=0);
						}
					current_from=current_from->_next_sibling;
					if(current_from!=last) {
						toit=append_child(parent(toit), current_from->data);
						}
					}
				}
			while(current_from!=last);

			return current_to;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::sibling_iterator tree<T, tree_nodeallocator>::replace(
			sibling_iterator orig_begin, 
			sibling_iterator orig_end, 
			sibling_iterator new_begin, 
			sibling_iterator new_end)
			{
			tree_node *orig_first=orig_begin.node;
			tree_node *new_first=new_begin.node;
			tree_node *orig_last=orig_first;
			while((++orig_begin)!=orig_end)
				orig_last=orig_last->_next_sibling;
			tree_node *new_last=new_first;
			while((++new_begin)!=new_end)
				new_last=new_last->_next_sibling;

			// insert all siblings in new_first..new_last before orig_first
			bool first=true;
			pre_order_iterator ret;
			while(1==1) {
				pre_order_iterator tt=insert_subtree(pre_order_iterator(orig_first), pre_order_iterator(new_first));
				if(first) {
					ret=tt;
					first=false;
					}
				if(new_first==new_last)
					break;
				new_first=new_first->_next_sibling;
				}

			// erase old range of siblings
			bool last=false;
			tree_node *next=orig_first;
			while(1==1) {
				if(next==orig_last) 
					last=true;
				next=next->_next_sibling;
				erase((pre_order_iterator)orig_first);
				if(last) 
					break;
				orig_first=next;
				}
			return ret;
			}

		template <class T, class tree_nodeallocator>
		template <typename iter>
		iter tree<T, tree_nodeallocator>::flatten(iter position)
			{
			if(position.node->_first_child==0)
				return position;

			tree_node *tmp=position.node->_first_child;
			while(tmp) {
				tmp->_parent=position.node->_parent;
				tmp=tmp->_next_sibling;
				} 
			if(position.node->_next_sibling) {
				position.node->_last_child->_next_sibling=position.node->_next_sibling;
				position.node->_next_sibling->_prev_sibling=position.node->_last_child;
				}
			else {
				position.node->_parent->_last_child=position.node->_last_child;
				}
			position.node->_next_sibling=position.node->_first_child;
			position.node->_next_sibling->_prev_sibling=position.node;
			position.node->_first_child=0;
			position.node->_last_child=0;

			return position;
			}


		template <class T, class tree_nodeallocator>
		template <typename iter>
		iter tree<T, tree_nodeallocator>::reparent(iter position, sibling_iterator begin, sibling_iterator end)
			{
			tree_node *first=begin.node;
			tree_node *last=first;

			assert(first!=position.node);

			if(begin==end) return begin;
			// determine last node
			while((++begin)!=end) {
				last=last->_next_sibling;
				}
			// move subtree
			if(first->_prev_sibling==0) {
				first->_parent->_first_child=last->_next_sibling;
				}
			else {
				first->_prev_sibling->_next_sibling=last->_next_sibling;
				}
			if(last->_next_sibling==0) {
				last->_parent->_last_child=first->_prev_sibling;
				}
			else {
				last->_next_sibling->_prev_sibling=first->_prev_sibling;
				}
			if(position.node->_first_child==0) {
				position.node->_first_child=first;
				position.node->_last_child=last;
				first->_prev_sibling=0;
				}
			else {
				position.node->_last_child->_next_sibling=first;
				first->_prev_sibling=position.node->_last_child;
				position.node->_last_child=last;
				}
			last->_next_sibling=0;

			tree_node *pos=first;
		   for(;;) {
				pos->_parent=position.node;
				if(pos==last) break;
				pos=pos->_next_sibling;
				}

			return first;
			}

		template <class T, class tree_nodeallocator>
		template <typename iter> iter tree<T, tree_nodeallocator>::reparent(iter position, iter from)
			{
			if(from.node->_first_child==0) return position;
			return reparent(position, from.node->_first_child, end(from));
			}

		template <class T, class tree_nodeallocator>
		template <typename iter> iter tree<T, tree_nodeallocator>::wrap(iter position, const T& x)
			{
			assert(position.node!=0);
			sibling_iterator fr=position, to=position;
			++to;
			iter ret = insert(position, x);
			reparent(ret, fr, to);
			return ret;
			}

		template <class T, class tree_nodeallocator>
		template <typename iter> iter tree<T, tree_nodeallocator>::wrap(iter from, iter to, const T& x)
			{
			assert(from.node!=0);
			iter ret = insert(from, x);
			reparent(ret, from, to);
			return ret;
			}

		template <class T, class tree_nodeallocator>
		template <typename iter> iter tree<T, tree_nodeallocator>::move_after(iter target, iter source)
		   {
		   tree_node *dst=target.node;
		   tree_node *src=source.node;
		   assert(dst);
		   assert(src);

		   if(dst==src) return source;
			if(dst->_next_sibling)
				if(dst->_next_sibling==src) // already in the right spot
					return source;

		   // take src out of the tree
		   if(src->_prev_sibling!=0) src->_prev_sibling->_next_sibling=src->_next_sibling;
		   else                     src->_parent->_first_child=src->_next_sibling;
		   if(src->_next_sibling!=0) src->_next_sibling->_prev_sibling=src->_prev_sibling;
		   else                     src->_parent->_last_child=src->_prev_sibling;

		   // connect it to the new point
		   if(dst->_next_sibling!=0) dst->_next_sibling->_prev_sibling=src;
		   else                     dst->_parent->_last_child=src;
		   src->_next_sibling=dst->_next_sibling;
		   dst->_next_sibling=src;
		   src->_prev_sibling=dst;
		   src->_parent=dst->_parent;
		   return src;
		   }

		template <class T, class tree_nodeallocator>
		template <typename iter> iter tree<T, tree_nodeallocator>::move_before(iter target, iter source)
		   {
		   tree_node *dst=target.node;
		   tree_node *src=source.node;
		   assert(dst);
		   assert(src);

		   if(dst==src) return source;
			if(dst->_prev_sibling)
				if(dst->_prev_sibling==src) // already in the right spot
					return source;

		   // take src out of the tree
		   if(src->_prev_sibling!=0) src->_prev_sibling->_next_sibling=src->_next_sibling;
		   else                     src->_parent->_first_child=src->_next_sibling;
		   if(src->_next_sibling!=0) src->_next_sibling->_prev_sibling=src->_prev_sibling;
		   else                     src->_parent->_last_child=src->_prev_sibling;

		   // connect it to the new point
		   if(dst->_prev_sibling!=0) dst->_prev_sibling->_next_sibling=src;
		   else                     dst->_parent->_first_child=src;
		   src->_prev_sibling=dst->_prev_sibling;
		   dst->_prev_sibling=src;
		   src->_next_sibling=dst;
		   src->_parent=dst->_parent;
		   return src;
		   }

		// specialisation for sibling_iterators
		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::sibling_iterator tree<T, tree_nodeallocator>::move_before(sibling_iterator target, 
																															  sibling_iterator source)
			{
			tree_node *dst=target.node;
			tree_node *src=source.node;
			tree_node *dst_prev_sibling;
			if(dst==0) { // must then be an end iterator
				dst_prev_sibling=target.parent_->_last_child;
				assert(dst_prev_sibling);
				}
			else dst_prev_sibling=dst->_prev_sibling;
			assert(src);

			if(dst==src) return source;
			if(dst_prev_sibling)
				if(dst_prev_sibling==src) // already in the right spot
					return source;

			// take src out of the tree
			if(src->_prev_sibling!=0) src->_prev_sibling->_next_sibling=src->_next_sibling;
			else                     src->_parent->_first_child=src->_next_sibling;
			if(src->_next_sibling!=0) src->_next_sibling->_prev_sibling=src->_prev_sibling;
			else                     src->_parent->_last_child=src->_prev_sibling;

			// connect it to the new point
			if(dst_prev_sibling!=0) dst_prev_sibling->_next_sibling=src;
			else                    target.parent_->_first_child=src;
			src->_prev_sibling=dst_prev_sibling;
			if(dst) {
				dst->_prev_sibling=src;
				src->_parent=dst->_parent;
				}
			else {
				src->_parent=dst_prev_sibling->_parent;
				}
			src->_next_sibling=dst;
			return src;
			}

		template <class T, class tree_nodeallocator>
		template <typename iter> iter tree<T, tree_nodeallocator>::move_ontop(iter target, iter source)
			{
			tree_node *dst=target.node;
			tree_node *src=source.node;
			assert(dst);
			assert(src);

			if(dst==src) return source;

		//	if(dst==src->_prev_sibling) {
		//
		//		}

			// remember connection points
			tree_node *b_prev_sibling=dst->_prev_sibling;
			tree_node *b_next_sibling=dst->_next_sibling;
			tree_node *b_parent=dst->_parent;

			// remove target
			erase(target);

			// take src out of the tree
			if(src->_prev_sibling!=0) src->_prev_sibling->_next_sibling=src->_next_sibling;
			else {
				assert(src->_parent!=0);
				src->_parent->_first_child=src->_next_sibling;
				}
			if(src->_next_sibling!=0) src->_next_sibling->_prev_sibling=src->_prev_sibling;
			else {
				assert(src->_parent!=0);
				src->_parent->_last_child=src->_prev_sibling;
				}

			// connect it to the new point
			if(b_prev_sibling!=0) b_prev_sibling->_next_sibling=src;
			else {
				assert(b_parent!=0);
				b_parent->_first_child=src;
				}
			if(b_next_sibling!=0) b_next_sibling->_prev_sibling=src;
			else {
				assert(b_parent!=0);
				b_parent->_last_child=src;
				}
			src->_prev_sibling=b_prev_sibling;
			src->_next_sibling=b_next_sibling;
			src->_parent=b_parent;
			return src;
			}


		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator> tree<T, tree_nodeallocator>::move_out(iterator source)
			{
			tree ret;

			// Move source node into the 'ret' tree.
			ret.head->_next_sibling = source.node;
			ret.feet->_prev_sibling = source.node;
			source.node->_parent=0;

			// Close the links in the current tree.
			if(source.node->_prev_sibling!=0) 
				source.node->_prev_sibling->_next_sibling = source.node->_next_sibling;

			if(source.node->_next_sibling!=0) 
				source.node->_next_sibling->_prev_sibling = source.node->_prev_sibling;

			// Fix source prev/next links.
			source.node->_prev_sibling = ret.head;
			source.node->_next_sibling = ret.feet;

			return ret; // A good compiler will move this, not copy.
			}

		template <class T, class tree_nodeallocator>
		template<typename iter> iter tree<T, tree_nodeallocator>::move_in(iter loc, tree& other)
			{
			if(other.head->_next_sibling==other.feet) return loc; // other tree is empty

			tree_node *other_first_head = other.head->_next_sibling;
			tree_node *other_last_head  = other.feet->_prev_sibling;

			sibling_iterator prev(loc);
			--prev;

			prev.node->_next_sibling = other_first_head;
			loc.node->_prev_sibling  = other_last_head;
			other_first_head->_prev_sibling = prev.node;
			other_last_head->_next_sibling  = loc.node;

			// Adjust parent pointers.
			tree_node *walk=other_first_head;
			while(true) {
				walk->_parent=loc.node->_parent;
				if(walk==other_last_head)
					break;
				walk=walk->_next_sibling;
				}

			// Close other tree.
			other.head->_next_sibling=other.feet;
			other.feet->_prev_sibling=other.head;

			return other_first_head;
			}

		template <class T, class tree_nodeallocator>
		template<typename iter> iter tree<T, tree_nodeallocator>::move_in_below(iter loc, tree& other)
			{
			if(other.head->_next_sibling==other.feet) return loc; // other tree is empty

			auto n = other.number_of_children(loc);
			return move_in_as_nth_child(loc, n, other);
			}

		template <class T, class tree_nodeallocator>
		template<typename iter> iter tree<T, tree_nodeallocator>::move_in_as_nth_child(iter loc, size_t n, tree& other)
			{
			if(other.head->_next_sibling==other.feet) return loc; // other tree is empty

			tree_node *other_first_head = other.head->_next_sibling;
			tree_node *other_last_head  = other.feet->_prev_sibling;

			if(n==0) {
				if(loc.node->_first_child==0) {
					loc.node->_first_child=other_first_head;
					loc.node->_last_child=other_last_head;
					other_last_head->_next_sibling=0;
					other_first_head->_prev_sibling=0;
					} 
				else {
					loc.node->_first_child->_prev_sibling=other_last_head;
					other_last_head->_next_sibling=loc.node->_first_child;
					loc.node->_first_child=other_first_head;
					other_first_head->_prev_sibling=0;
					}
				}
			else {
				--n;
				tree_node *walk = loc.node->_first_child;
				while(true) {
					if(walk==0)
						throw std::range_error("tree: move_in_as_nth_child position out of range");
					if(n==0) 
						break;
					--n;
					walk = walk->_next_sibling;
					}
				if(walk->_next_sibling==0) 
					loc.node->_last_child=other_last_head;
				else 
					walk->_next_sibling->_prev_sibling=other_last_head;
				other_last_head->_next_sibling=walk->_next_sibling;
				walk->_next_sibling=other_first_head;
				other_first_head->_prev_sibling=walk;
				}

			// Adjust parent pointers.
			tree_node *walk=other_first_head;
			while(true) {
				walk->_parent=loc.node;
				if(walk==other_last_head)
					break;
				walk=walk->_next_sibling;
				}

			// Close other tree.
			other.head->_next_sibling=other.feet;
			other.feet->_prev_sibling=other.head;

			return other_first_head;
			}


		template <class T, class tree_nodeallocator>
		void tree<T, tree_nodeallocator>::merge(sibling_iterator to1,   sibling_iterator to2,
																sibling_iterator from1, sibling_iterator from2,
																bool duplicate_leaves)
			{
			sibling_iterator fnd;
			while(from1!=from2) {
				if((fnd=std::find(to1, to2, (*from1))) != to2) { // element found
					if(from1.begin()==from1.end()) { // full depth reached
						if(duplicate_leaves)
							append_child(parent(to1), (*from1));
						}
					else { // descend further
						merge(fnd.begin(), fnd.end(), from1.begin(), from1.end(), duplicate_leaves);
						}
					}
				else { // element missing
					insert_subtree(to2, from1);
					}
				++from1;
				}
			}

		template <class T, class tree_nodeallocator>
		void tree<T, tree_nodeallocator>::merge(iterator to, iterator from, bool duplicate_leaves)
			{
			sibling_iterator to1(to);
			sibling_iterator to2=to1;
			++to2;
			sibling_iterator from1(from);
			sibling_iterator from2=from1;
			++from2;

			merge(to1, to2, from1, from2, duplicate_leaves);
			}


		template <class T, class tree_nodeallocator>
		void tree<T, tree_nodeallocator>::sort(sibling_iterator from, sibling_iterator to, bool deep)
			{
			std::less<T> comp;
			sort(from, to, comp, deep);
			}

		template <class T, class tree_nodeallocator>
		template <class StrictWeakOrdering>
		void tree<T, tree_nodeallocator>::sort(sibling_iterator from, sibling_iterator to, 
															 StrictWeakOrdering comp, bool deep)
			{
			if(from==to) return;
			// make list of sorted nodes
			// CHECK: if multiset stores equivalent nodes in the order in which they
			// are inserted, then this routine should be called 'stable_sort'.
			std::multiset<tree_node *, compare_nodes<StrictWeakOrdering> > nodes(comp);
			sibling_iterator it=from, it2=to;
			while(it != to) {
				nodes.insert(it.node);
				++it;
				}
			// reassemble
			--it2;

			// prev and next are the nodes before and after the sorted range
			tree_node *prev=from.node->_prev_sibling;
			tree_node *next=it2.node->_next_sibling;
			typename std::multiset<tree_node *, compare_nodes<StrictWeakOrdering> >::iterator nit=nodes.begin(), eit=nodes.end();
			if(prev==0) {
				if((*nit)->_parent!=0) // to catch "sorting the head" situations, when there is no parent
					(*nit)->_parent->_first_child=(*nit);
				}
			else prev->_next_sibling=(*nit);

			--eit;
			while(nit!=eit) {
				(*nit)->_prev_sibling=prev;
				if(prev)
					prev->_next_sibling=(*nit);
				prev=(*nit);
				++nit;
				}
			// prev now points to the last-but-one node in the sorted range
			if(prev)
				prev->_next_sibling=(*eit);

			// eit points to the last node in the sorted range.
			(*eit)->_next_sibling=next;
		   (*eit)->_prev_sibling=prev; // missed in the loop above
			if(next==0) {
				if((*eit)->_parent!=0) // to catch "sorting the head" situations, when there is no parent
					(*eit)->_parent->_last_child=(*eit);
				}
			else next->_prev_sibling=(*eit);

			if(deep) {	// sort the children of each node too
				sibling_iterator bcs(*nodes.begin());
				sibling_iterator ecs(*eit);
				++ecs;
				while(bcs!=ecs) {
					sort(begin(bcs), end(bcs), comp, deep);
					++bcs;
					}
				}
			}

		template <class T, class tree_nodeallocator>
		template <typename iter>
		bool tree<T, tree_nodeallocator>::equal(const iter& one_, const iter& two, const iter& three_) const
			{
			std::equal_to<T> comp;
			return equal(one_, two, three_, comp);
			}

		template <class T, class tree_nodeallocator>
		template <typename iter>
		bool tree<T, tree_nodeallocator>::equal_subtree(const iter& one_, const iter& two_) const
			{
			std::equal_to<T> comp;
			return equal_subtree(one_, two_, comp);
			}

		template <class T, class tree_nodeallocator>
		template <typename iter, class BinaryPredicate>
		bool tree<T, tree_nodeallocator>::equal(const iter& one_, const iter& two, const iter& three_, BinaryPredicate fun) const
			{
			pre_order_iterator one(one_), three(three_);

		//	if(one==two && is_valid(three) && three.number_of_children()!=0)
		//		return false;
			while(one!=two && is_valid(three)) {
				if(!fun(*one,*three))
					return false;
				if(one.number_of_children()!=three.number_of_children()) 
					return false;
				++one;
				++three;
				}
			return true;
			}

		template <class T, class tree_nodeallocator>
		template <typename iter, class BinaryPredicate>
		bool tree<T, tree_nodeallocator>::equal_subtree(const iter& one_, const iter& two_, BinaryPredicate fun) const
			{
			pre_order_iterator one(one_), two(two_);

			if(!fun(*one,*two)) return false;
			if(number_of_children(one)!=number_of_children(two)) return false;
			return equal(begin(one),end(one),begin(two),fun);
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator> tree<T, tree_nodeallocator>::subtree(sibling_iterator from, sibling_iterator to) const
			{
			assert(from!=to); // if from==to, the range is empty, hence no tree to return.

			tree tmp;
			tmp.set_head(value_type());
			tmp.replace(tmp.begin(), tmp.end(), from, to);
			return tmp;
			}

		template <class T, class tree_nodeallocator>
		void tree<T, tree_nodeallocator>::subtree(tree& tmp, sibling_iterator from, sibling_iterator to) const
			{
			assert(from!=to); // if from==to, the range is empty, hence no tree to return.

			tmp.set_head(value_type());
			tmp.replace(tmp.begin(), tmp.end(), from, to);
			}

		template <class T, class tree_nodeallocator>
		size_t tree<T, tree_nodeallocator>::size() const {
			size_t i=0;
			pre_order_iterator it=begin(), eit=end();
			while(it!=eit) {
				++i;
				++it;
			}
			return i;
		}

		template <class T, class tree_nodeallocator>
		size_t tree<T, tree_nodeallocator>::size(const iterator_base& top) const {
			size_t i=0;
			pre_order_iterator it=top, eit=top;
			eit.skip_children();
			++eit;
			while(it!=eit) {
				++i;
				++it;
			}
			return i;
		}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::empty() const
			{
			pre_order_iterator it=begin(), eit=end();
			return (it==eit);
			}

		template <class T, class tree_nodeallocator>
		int tree<T, tree_nodeallocator>::depth(const iterator_base& it) {
			tree_node* pos=it.node;
			assert(pos!=0);
			int ret=0;
			while(pos->_parent!=0) {
				pos=pos->_parent;
				++ret;
			}
			return ret;
		}

		template <class T, class tree_nodeallocator>
		int tree<T, tree_nodeallocator>::depth(const iterator_base& it, const iterator_base& root) 
			{
			tree_node* pos=it.node;
			assert(pos!=0);
			int ret=0;
			while(pos->_parent!=0 && pos!=root.node) {
				pos=pos->_parent;
				++ret;
				}
			return ret;
			}

		template <class T, class tree_nodeallocator>
		template <class Predicate>
		int tree<T, tree_nodeallocator>::depth(const iterator_base& it, Predicate p) 
			{
			tree_node* pos=it.node;
			assert(pos!=0);
			int ret=0;
			while(pos->_parent!=0) {
				pos=pos->_parent;
				if(p(pos))
					++ret;
				}
			return ret;
			}

		template <class T, class tree_nodeallocator>
		template <class Predicate>
		int tree<T, tree_nodeallocator>::distance(const iterator_base& top, const iterator_base& bottom, Predicate p) 
			{
			tree_node* pos=bottom.node;
			assert(pos!=0);
			int ret=0;
			while(pos->_parent!=0 && pos!=top.node) {
				pos=pos->_parent;
				if(p(pos))
					++ret;
				}
			return ret;
			}

		template <class T, class tree_nodeallocator>
		int tree<T, tree_nodeallocator>::max_depth() const
			{
			int maxd=-1;
			for(tree_node *it = head->_next_sibling; it!=feet; it=it->_next_sibling)
				maxd=std::max(maxd, max_depth(it));

			return maxd;
			}


		template <class T, class tree_nodeallocator>
		int tree<T, tree_nodeallocator>::max_depth(const iterator_base& pos) const{
			tree_node *tmp = pos.node;

			if(tmp == 0 || tmp == head || tmp == feet) return -1;

			int curdepth = 0, maxdepth = 0;
			while(true) { // try to walk the bottom of the tree
				while(tmp->_first_child == 0) {
					if(tmp == pos.node) return maxdepth;
					if(tmp->_next_sibling == 0) {
						// try to walk up and then right again
						do {
							tmp = tmp->_parent;
		               		if(tmp == 0) return maxdepth;
		               		--curdepth;
						}
						while(tmp->_next_sibling == 0);
					}
		         if(tmp == pos.node) return maxdepth;
					tmp = tmp->_next_sibling;
				}
				tmp = tmp->_first_child;
				++curdepth;
				maxdepth = std::max(curdepth, maxdepth);
			} 
		}

		template <class T, class tree_nodeallocator>
		unsigned int tree<T, tree_nodeallocator>::number_of_children(const iterator_base& it) 
		{
			tree_node *pos=it.node->_first_child;
			if(pos==0) return 0;

			unsigned int ret=1;
		//	  while(pos!=it.node->_last_child) {
		//		  ++ret;
		//		  pos=pos->_next_sibling;
		//		  }
			while((pos=pos->_next_sibling))
				++ret;
			return ret;
		}

		template <class T, class tree_nodeallocator>
		unsigned int tree<T, tree_nodeallocator>::number_of_siblings(const iterator_base& it) const
			{
			tree_node *pos=it.node;
			unsigned int ret=0;
			// count forward
			while(pos->_next_sibling && 
					pos->_next_sibling!=head &&
					pos->_next_sibling!=feet) {
				++ret;
				pos=pos->_next_sibling;
				}
			// count backward
			pos=it.node;
			while(pos->_prev_sibling && 
					pos->_prev_sibling!=head &&
					pos->_prev_sibling!=feet) {
				++ret;
				pos=pos->_prev_sibling;
				}

			return ret;
			}

		template <class T, class tree_nodeallocator>
		void tree<T, tree_nodeallocator>::swap(sibling_iterator it)
			{
			tree_node *nxt=it.node->_next_sibling;
			if(nxt) {
				if(it.node->_prev_sibling)
					it.node->_prev_sibling->_next_sibling=nxt;
				else
					it.node->_parent->_first_child=nxt;
				nxt->_prev_sibling=it.node->_prev_sibling;
				tree_node *nxtnxt=nxt->_next_sibling;
				if(nxtnxt)
					nxtnxt->_prev_sibling=it.node;
				else
					it.node->_parent->_last_child=it.node;
				nxt->_next_sibling=it.node;
				it.node->_prev_sibling=nxt;
				it.node->_next_sibling=nxtnxt;
				}
			}

		template <class T, class tree_nodeallocator>
		void tree<T, tree_nodeallocator>::swap(iterator one, iterator two)
			{
			// if one and two are adjacent siblings, use the sibling swap
			if(one.node->_next_sibling==two.node) swap(one);
			else if(two.node->_next_sibling==one.node) swap(two);
			else {
				tree_node *nxt1=one.node->_next_sibling;
				tree_node *nxt2=two.node->_next_sibling;
				tree_node *pre1=one.node->_prev_sibling;
				tree_node *pre2=two.node->_prev_sibling;
				tree_node *par1=one.node->_parent;
				tree_node *par2=two.node->_parent;

				// reconnect
				one.node->_parent=par2;
				one.node->_next_sibling=nxt2;
				if(nxt2) nxt2->_prev_sibling=one.node;
				else     par2->_last_child=one.node;
				one.node->_prev_sibling=pre2;
				if(pre2) pre2->_next_sibling=one.node;
				else     par2->_first_child=one.node;    

				two.node->_parent=par1;
				two.node->_next_sibling=nxt1;
				if(nxt1) nxt1->_prev_sibling=two.node;
				else     par1->_last_child=two.node;
				two.node->_prev_sibling=pre1;
				if(pre1) pre1->_next_sibling=two.node;
				else     par1->_first_child=two.node;
				}
			}

		// template <class BinaryPredicate>
		// tree<T, tree_nodeallocator>::iterator tree<T, tree_nodeallocator>::find_subtree(
		// 	sibling_iterator subfrom, sibling_iterator subto, iterator from, iterator to, 
		// 	BinaryPredicate fun) const
		// 	{
		// 	assert(1==0); // this routine is not finished yet.
		// 	while(from!=to) {
		// 		if(fun(*subfrom, *from)) {
		// 			
		// 			}
		// 		}
		// 	return to;
		// 	}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::is_in_subtree(const iterator_base& it, const iterator_base& top) const
		   {
			sibling_iterator first=top;
			sibling_iterator last=first;
			++last;
			return is_in_subtree(it, first, last);
			}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::is_in_subtree(const iterator_base& it, const iterator_base& begin, 
																		 const iterator_base& end) const
			{
			// FIXME: this should be optimised.
			pre_order_iterator tmp=begin;
			while(tmp!=end) {
				if(tmp==it) return true;
				++tmp;
				}
			return false;
			}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::is_valid(const iterator_base& it) const
			{
			if(it.node==0 || it.node==feet || it.node==head) return false;
			else return true;
			}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::is_head(const iterator_base& it) 
		  	{
			if(it.node->_parent==0) return true;
			return false;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::iterator tree<T, tree_nodeallocator>::lowest_common_ancestor(
			const iterator_base& one, const iterator_base& two) const
			{
			std::set<iterator, iterator_base_less> parents;

			// Walk up from 'one' storing all parents.
			iterator walk=one;
			do {
				walk=parent(walk);
				parents.insert(walk);
				}
			while( walk.node->_parent );

			// Walk up from 'two' until we encounter a node in parents.
			walk=two;
			do {
				walk=parent(walk);
				if(parents.find(walk) != parents.end()) break;
				}
			while( walk.node->_parent );

			return walk;
			}

		template <class T, class tree_nodeallocator>
		unsigned int tree<T, tree_nodeallocator>::index(sibling_iterator it) const
			{
			unsigned int ind=0;
			if(it.node->_parent==0) {
				while(it.node->_prev_sibling!=head) {
					it.node=it.node->_prev_sibling;
					++ind;
					}
				}
			else {
				while(it.node->_prev_sibling!=0) {
					it.node=it.node->_prev_sibling;
					++ind;
					}
				}
			return ind;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::sibling_iterator tree<T, tree_nodeallocator>::sibling(const iterator_base& it, unsigned int num) const
		   {
		   tree_node *tmp;
		   if(it.node->_parent==0) {
		      tmp=head->_next_sibling;
		      while(num) {
		         tmp = tmp->_next_sibling;
		         --num;
		         }
		      }
		   else {
		      tmp=it.node->_parent->_first_child;
		      while(num) {
		         assert(tmp!=0);
		         tmp = tmp->_next_sibling;
		         --num;
		         }
		      }
		   return tmp;
		   }

		template <class T, class tree_nodeallocator>
		void tree<T, tree_nodeallocator>::debug_verify_consistency() const
			{
			iterator it=begin();
			while(it!=end()) {
				// std::cerr << *it << " (" << it.node << ")" << std::endl;
				if(it.node->_parent!=0) {
					if(it.node->_prev_sibling==0) 
						assert(it.node->_parent->_first_child==it.node);
					else 
						assert(it.node->_prev_sibling->_next_sibling==it.node);
					if(it.node->_next_sibling==0) 
						assert(it.node->_parent->_last_child==it.node);
					else
						assert(it.node->_next_sibling->_prev_sibling==it.node);
					}
				++it;
				}
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::sibling_iterator tree<T, tree_nodeallocator>::child(const iterator_base& it, unsigned int num) 
			{
			tree_node *tmp=it.node->_first_child;
			while(num--) {
				assert(tmp!=0);
				tmp=tmp->_next_sibling;
				}
			return tmp;
			}




		// Iterator base

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::iterator_base::iterator_base()
			: node(0), skip_current_children_(false)
			{
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::iterator_base::iterator_base(tree_node *tn)
			: node(tn), skip_current_children_(false)
			{
			}

		template <class T, class tree_nodeallocator>
		T& tree<T, tree_nodeallocator>::iterator_base::operator*() const
			{
			return node->data;
			}

		template <class T, class tree_nodeallocator>
		T* tree<T, tree_nodeallocator>::iterator_base::operator->() const
			{
			return &(node->data);
			}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::post_order_iterator::operator!=(const post_order_iterator& other) const
			{
			if(other.node!=this->node) return true;
			else return false;
			}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::post_order_iterator::operator==(const post_order_iterator& other) const
			{
			if(other.node==this->node) return true;
			else return false;
			}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::pre_order_iterator::operator!=(const pre_order_iterator& other) const
			{
			if(other.node!=this->node) return true;
			else return false;
			}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::pre_order_iterator::operator==(const pre_order_iterator& other) const
			{
			if(other.node==this->node) return true;
			else return false;
			}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::sibling_iterator::operator!=(const sibling_iterator& other) const
			{
			if(other.node!=this->node) return true;
			else return false;
			}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::sibling_iterator::operator==(const sibling_iterator& other) const
			{
			if(other.node==this->node) return true;
			else return false;
			}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::leaf_iterator::operator!=(const leaf_iterator& other) const
		   {
		   if(other.node!=this->node) return true;
		   else return false;
		   }

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::leaf_iterator::operator==(const leaf_iterator& other) const
		   {
		   if(other.node==this->node && other.top_node==this->top_node) return true;
		   else return false;
		   }

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::sibling_iterator tree<T, tree_nodeallocator>::iterator_base::begin() const
			{
			if(node->_first_child==0) 
				return end();

			sibling_iterator ret(node->_first_child);
			ret.parent_=this->node;
			return ret;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::sibling_iterator tree<T, tree_nodeallocator>::iterator_base::end() const
			{
			sibling_iterator ret(0);
			ret.parent_=node;
			return ret;
			}

		template <class T, class tree_nodeallocator>
		void tree<T, tree_nodeallocator>::iterator_base::skip_children()
			{
			skip_current_children_=true;
			}

		template <class T, class tree_nodeallocator>
		void tree<T, tree_nodeallocator>::iterator_base::skip_children(bool skip)
		   {
		   skip_current_children_=skip;
		   }

		template <class T, class tree_nodeallocator>
		unsigned int tree<T, tree_nodeallocator>::iterator_base::number_of_children() const
			{
			tree_node *pos=node->_first_child;
			if(pos==0) return 0;

			unsigned int ret=1;
			while(pos!=node->_last_child) {
				++ret;
				pos=pos->_next_sibling;
				}
			return ret;
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::pre_order_iterator::pre_order_iterator() 
			: iterator_base(0)
			{
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::pre_order_iterator::pre_order_iterator(tree_node *tn)
			: iterator_base(tn)
			{
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::pre_order_iterator::pre_order_iterator(const iterator_base &other)
			: iterator_base(other.node)
			{
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::pre_order_iterator::pre_order_iterator(const sibling_iterator& other)
			: iterator_base(other.node)
			{
			if(this->node==0) {
				if(other.range_last()!=0)
					this->node=other.range_last();
				else 
					this->node=other.parent_;
				this->skip_children();
				++(*this);
				}
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::pre_order_iterator& tree<T, tree_nodeallocator>::pre_order_iterator::operator++() {
			assert(this->node!=0);
			if(!this->skip_current_children_ && this->node->_first_child != 0) { // 如果当前节点有子节点，同时状态为<不跳过下一个子节点>
				this->node=this->node->_first_child; // 将当前节点设置为当前节点的子节点
				}
			else { // 状态为<跳过子节点以及子节点子树> 或者为没有子节点，反正就是不会查询到子节点
				this->skip_current_children_=false;	// 直接重置状态，等待下一次设置
				while(this->node->_next_sibling==0) { // 如果当前节点没有下一个兄弟节点，将当前节点设置为当前节点的父节点，向上查询，
				//指导查到有下一个兄弟节点的节点，或者直到根节点，停止
					this->node=this->node->_parent;
					if(this->node==0)
						return *this;
					}
				this->node=this->node->_next_sibling; // 如果当前节点有下一个兄弟节点，当前节点设置为当前节点的下一个兄弟节点
				}
			return *this;
		}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::pre_order_iterator& tree<T, tree_nodeallocator>::pre_order_iterator::operator--()
			{
			assert(this->node!=0);
			if(this->node->_prev_sibling) {
				this->node=this->node->_prev_sibling;
				while(this->node->_last_child)
					this->node=this->node->_last_child;
				}
			else {
				this->node=this->node->_parent;
				if(this->node==0)
					return *this;
				}
			return *this;
		}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::pre_order_iterator tree<T, tree_nodeallocator>::pre_order_iterator::operator++(int)
			{
			pre_order_iterator copy = *this;
			++(*this);
			return copy;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::pre_order_iterator& tree<T, tree_nodeallocator>::pre_order_iterator::next_skip_children() {
			(*this).skip_children();
			(*this)++;
			return *this;
		}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::pre_order_iterator tree<T, tree_nodeallocator>::pre_order_iterator::operator--(int)
		{
		  pre_order_iterator copy = *this;
		  --(*this);
		  return copy;
		}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::pre_order_iterator& tree<T, tree_nodeallocator>::pre_order_iterator::operator+=(unsigned int num)
			{
			while(num>0) {
				++(*this);
				--num;
				}
			return (*this);
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::pre_order_iterator& tree<T, tree_nodeallocator>::pre_order_iterator::operator-=(unsigned int num)
			{
			while(num>0) {
				--(*this);
				--num;
				}
			return (*this);
			}

		// Post-order iterator
		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::post_order_iterator::post_order_iterator() 
			: iterator_base(0)
			{
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::post_order_iterator::post_order_iterator(tree_node *tn)
			: iterator_base(tn)
			{
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::post_order_iterator::post_order_iterator(const iterator_base &other)
			: iterator_base(other.node)
			{
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::post_order_iterator::post_order_iterator(const sibling_iterator& other)
			: iterator_base(other.node)
			{
			if(this->node==0) {
				if(other.range_last()!=0)
					this->node=other.range_last();
				else 
					this->node=other.parent_;
				this->skip_children();
				++(*this);
				}
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::post_order_iterator& tree<T, tree_nodeallocator>::post_order_iterator::operator++(){
			assert(this->node != 0);
			if(this->node->_next_sibling == 0) { // 如果当前节点没有兄弟节点了，那么将当前节点置为当前节点的父节点，往上回溯
				this->node = this->node->_parent;
				this->skip_current_children_ = false;
			}
			else { // 如果当前节点存在兄弟节点，那么
				this->node = this->node->_next_sibling;
				if(this->skip_current_children_) { // 如果状态<跳过子节点子树>成立，则
					this->skip_current_children_ = false;
				}
				else {
					while(this->node->_first_child) // 如果当前节点存在子节点，那么往下查询指导获得叶子结点
						this->node = this->node->_first_child;
				}
			}
			return *this;
		}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::post_order_iterator& tree<T, tree_nodeallocator>::post_order_iterator::operator--() {
			assert(this->node != 0);
			if(this->skip_current_children_ || this->node->_last_child == 0) { // 当状态<跳过下一个节点>成立，或者当前节点没有兄弟节点
				this->skip_current_children_ = false;
				while(this->node->_prev_sibling == 0)
					this->node = this->node->_parent;
				this->node = this->node->_prev_sibling;
			}
			else {
				this->node = this->node->_last_child;
			}
			return *this;
		}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::post_order_iterator tree<T, tree_nodeallocator>::post_order_iterator::operator++(int)
			{
			post_order_iterator copy = *this;
			++(*this);
			return copy;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::post_order_iterator tree<T, tree_nodeallocator>::post_order_iterator::operator--(int)
			{
			post_order_iterator copy = *this;
			--(*this);
			return copy;
			}


		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::post_order_iterator& tree<T, tree_nodeallocator>::post_order_iterator::operator+=(unsigned int num)
			{
			while(num>0) {
				++(*this);
				--num;
				}
			return (*this);
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::post_order_iterator& tree<T, tree_nodeallocator>::post_order_iterator::operator-=(unsigned int num)
			{
			while(num>0) {
				--(*this);
				--num;
				}
			return (*this);
			}

		template <class T, class tree_nodeallocator>
		void tree<T, tree_nodeallocator>::post_order_iterator::descend_all()
			{
			assert(this->node!=0);
			while(this->node->_first_child)
				this->node=this->node->_first_child;
			}


		// Breadth-first iterator

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::breadth_first_queued_iterator::breadth_first_queued_iterator()
			: iterator_base()
			{
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::breadth_first_queued_iterator::breadth_first_queued_iterator(tree_node *tn)
			: iterator_base(tn)
			{
			traversal_queue.push(tn);
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::breadth_first_queued_iterator::breadth_first_queued_iterator(const iterator_base& other)
			: iterator_base(other.node)
			{
			traversal_queue.push(other.node);
			}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::breadth_first_queued_iterator::operator!=(const breadth_first_queued_iterator& other) const
			{
			if(other.node!=this->node) return true;
			else return false;
			}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::breadth_first_queued_iterator::operator==(const breadth_first_queued_iterator& other) const
			{
			if(other.node==this->node) return true;
			else return false;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::breadth_first_queued_iterator& tree<T, tree_nodeallocator>::breadth_first_queued_iterator::operator++()
			{
			assert(this->node!=0);

			// Add child nodes and pop current node
			sibling_iterator sib=this->begin();
			while(sib!=this->end()) {
				traversal_queue.push(sib.node);
				++sib;
				}
			traversal_queue.pop();
			if(traversal_queue.size()>0)
				this->node=traversal_queue.front();
			else 
				this->node=0;
			return (*this);
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::breadth_first_queued_iterator tree<T, tree_nodeallocator>::breadth_first_queued_iterator::operator++(int)
			{
			breadth_first_queued_iterator copy = *this;
			++(*this);
			return copy;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::breadth_first_queued_iterator& tree<T, tree_nodeallocator>::breadth_first_queued_iterator::operator+=(unsigned int num)
			{
			while(num>0) {
				++(*this);
				--num;
				}
			return (*this);
			}



		// Fixed depth iterator

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::fixed_depth_iterator::fixed_depth_iterator()
			: iterator_base()
			{
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::fixed_depth_iterator::fixed_depth_iterator(tree_node *tn)
			: iterator_base(tn), top_node(0)
			{
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::fixed_depth_iterator::fixed_depth_iterator(const iterator_base& other)
			: iterator_base(other.node), top_node(0)
			{
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::fixed_depth_iterator::fixed_depth_iterator(const sibling_iterator& other)
			: iterator_base(other.node), top_node(0)
			{
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::fixed_depth_iterator::fixed_depth_iterator(const fixed_depth_iterator& other)
			: iterator_base(other.node), top_node(other.top_node)
			{
			}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::fixed_depth_iterator::operator==(const fixed_depth_iterator& other) const
			{
			if(other.node==this->node && other.top_node==top_node) return true;
			else return false;
			}

		template <class T, class tree_nodeallocator>
		bool tree<T, tree_nodeallocator>::fixed_depth_iterator::operator!=(const fixed_depth_iterator& other) const
			{
			if(other.node!=this->node || other.top_node!=top_node) return true;
			else return false;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::fixed_depth_iterator& tree<T, tree_nodeallocator>::fixed_depth_iterator::operator++()
			{
			assert(this->node!=0);

			if(this->node->_next_sibling) {
				this->node=this->node->_next_sibling;
				}
			else { 
				int relative_depth=0;
			   upper:
				do {
					if(this->node==this->top_node) {
						this->node=0; // FIXME: return a proper fixed_depth end iterator once implemented
						return *this;
						}
					this->node=this->node->_parent;
					if(this->node==0) return *this;
					--relative_depth;
					} while(this->node->_next_sibling==0);
			   lower:
				this->node=this->node->_next_sibling;
				while(this->node->_first_child==0) {
					if(this->node->_next_sibling==0)
						goto upper;
					this->node=this->node->_next_sibling;
					if(this->node==0) return *this;
					}
				while(relative_depth<0 && this->node->_first_child!=0) {
					this->node=this->node->_first_child;
					++relative_depth;
					}
				if(relative_depth<0) {
					if(this->node->_next_sibling==0) goto upper;
					else                          goto lower;
					}
				}
			return *this;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::fixed_depth_iterator& tree<T, tree_nodeallocator>::fixed_depth_iterator::operator--()
			{
			assert(this->node!=0);

			if(this->node->_prev_sibling) {
				this->node=this->node->_prev_sibling;
				}
			else { 
				int relative_depth=0;
			   upper:
				do {
					if(this->node==this->top_node) {
						this->node=0;
						return *this;
						}
					this->node=this->node->_parent;
					if(this->node==0) return *this;
					--relative_depth;
					} while(this->node->_prev_sibling==0);
			   lower:
				this->node=this->node->_prev_sibling;
				while(this->node->_last_child==0) {
					if(this->node->_prev_sibling==0)
						goto upper;
					this->node=this->node->_prev_sibling;
					if(this->node==0) return *this;
					}
				while(relative_depth<0 && this->node->_last_child!=0) {
					this->node=this->node->_last_child;
					++relative_depth;
					}
				if(relative_depth<0) {
					if(this->node->_prev_sibling==0) goto upper;
					else                            goto lower;
					}
				}
			return *this;

		//
		//
		//	assert(this->node!=0);
		//	if(this->node->_prev_sibling!=0) {
		//		this->node=this->node->_prev_sibling;
		//		assert(this->node!=0);
		//		if(this->node->_parent==0 && this->node->_prev_sibling==0) // head element
		//			this->node=0;
		//		}
		//	else {
		//		tree_node *par=this->node->_parent;
		//		do {
		//			par=par->_prev_sibling;
		//			if(par==0) { // FIXME: need to keep track of this!
		//				this->node=0;
		//				return *this;
		//				}
		//			} while(par->_last_child==0);
		//		this->node=par->_last_child;
		//		}
		//	return *this;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::fixed_depth_iterator tree<T, tree_nodeallocator>::fixed_depth_iterator::operator++(int)
			{
			fixed_depth_iterator copy = *this;
			++(*this);
			return copy;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::fixed_depth_iterator tree<T, tree_nodeallocator>::fixed_depth_iterator::operator--(int)
		   {
			fixed_depth_iterator copy = *this;
			--(*this);
			return copy;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::fixed_depth_iterator& tree<T, tree_nodeallocator>::fixed_depth_iterator::operator-=(unsigned int num)
			{
			while(num>0) {
				--(*this);
				--(num);
				}
			return (*this);
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::fixed_depth_iterator& tree<T, tree_nodeallocator>::fixed_depth_iterator::operator+=(unsigned int num)
			{
			while(num>0) {
				++(*this);
				--(num);
				}
			return *this;
			}


		// Sibling iterator

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::sibling_iterator::sibling_iterator() 
			: iterator_base()
			{
			set_parent_();
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::sibling_iterator::sibling_iterator(tree_node *tn)
			: iterator_base(tn)
			{
			set_parent_();
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::sibling_iterator::sibling_iterator(const iterator_base& other)
			: iterator_base(other.node)
			{
			set_parent_();
			}

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::sibling_iterator::sibling_iterator(const sibling_iterator& other)
			: iterator_base(other), parent_(other.parent_)
			{
			}

		template <class T, class tree_nodeallocator>
		void tree<T, tree_nodeallocator>::sibling_iterator::set_parent_()
			{
			parent_=0;
			if(this->node==0) return;
			if(this->node->_parent!=0)
				parent_=this->node->_parent;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::sibling_iterator& tree<T, tree_nodeallocator>::sibling_iterator::operator++()
			{
			if(this->node)
				this->node=this->node->_next_sibling;
			return *this;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::sibling_iterator& tree<T, tree_nodeallocator>::sibling_iterator::operator--()
			{
			if(this->node) this->node=this->node->_prev_sibling;
			else {
				assert(parent_);
				this->node=parent_->_last_child;
				}
			return *this;
		}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::sibling_iterator tree<T, tree_nodeallocator>::sibling_iterator::operator++(int)
			{
			sibling_iterator copy = *this;
			++(*this);
			return copy;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::sibling_iterator tree<T, tree_nodeallocator>::sibling_iterator::operator--(int)
			{
			sibling_iterator copy = *this;
			--(*this);
			return copy;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::sibling_iterator& tree<T, tree_nodeallocator>::sibling_iterator::operator+=(unsigned int num)
			{
			while(num>0) {
				++(*this);
				--num;
				}
			return (*this);
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::sibling_iterator& tree<T, tree_nodeallocator>::sibling_iterator::operator-=(unsigned int num)
			{
			while(num>0) {
				--(*this);
				--num;
				}
			return (*this);
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::tree_node *tree<T, tree_nodeallocator>::sibling_iterator::range_first() const
			{
			tree_node *tmp=parent_->_first_child;
			return tmp;
			}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::tree_node *tree<T, tree_nodeallocator>::sibling_iterator::range_last() const
			{
			return parent_->_last_child;
			}

		// Leaf iterator

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::leaf_iterator::leaf_iterator() 
		   : iterator_base(0), top_node(0)
		   {
		   }

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::leaf_iterator::leaf_iterator(tree_node *tn, tree_node *top)
		   : iterator_base(tn), top_node(top)
		   {
		   }

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::leaf_iterator::leaf_iterator(const iterator_base &other)
		   : iterator_base(other.node), top_node(0)
		   {
		   }

		template <class T, class tree_nodeallocator>
		tree<T, tree_nodeallocator>::leaf_iterator::leaf_iterator(const sibling_iterator& other)
		   : iterator_base(other.node), top_node(0)
		   {
		   if(this->node==0) {
		      if(other.range_last()!=0)
		         this->node=other.range_last();
		      else 
		         this->node=other.parent_;
		      ++(*this);
		      }
		   }

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::leaf_iterator& tree<T, tree_nodeallocator>::leaf_iterator::operator++(){
			assert(this->node!=0);
			if(this->node->_first_child!=0) { // current node is no longer leaf (children got added)
				 while(this->node->_first_child) 
					  this->node=this->node->_first_child;
				 }
			else {
				 while(this->node->_next_sibling==0) { 
					  if (this->node->_parent==0) return *this;
					  this->node=this->node->_parent;
					  if (top_node != 0 && this->node==top_node) return *this;
					  }
				 this->node=this->node->_next_sibling;
				 while(this->node->_first_child)
					  this->node=this->node->_first_child;
				 }
			return *this;
		}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::leaf_iterator& tree<T, tree_nodeallocator>::leaf_iterator::operator--(){
			assert(this->node!=0);
			while (this->node->_prev_sibling==0) {
				if (this->node->_parent==0) return *this;
				this->node=this->node->_parent;
				if (top_node !=0 && this->node==top_node) return *this; 
				}
			this->node=this->node->_prev_sibling;
			while(this->node->_last_child)
				this->node=this->node->_last_child;
			return *this;
		}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::leaf_iterator tree<T, tree_nodeallocator>::leaf_iterator::operator++(int){
		   leaf_iterator copy = *this;
		   ++(*this);
		   return copy;
		}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::leaf_iterator tree<T, tree_nodeallocator>::leaf_iterator::operator--(int){
		   leaf_iterator copy = *this;
		   --(*this);
		   return copy;
		}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::leaf_iterator& tree<T, tree_nodeallocator>::leaf_iterator::operator+=(unsigned int num){
		   while(num>0) {
		      ++(*this);
		      --num;
		      }
		   return (*this);
		}

		template <class T, class tree_nodeallocator>
		typename tree<T, tree_nodeallocator>::leaf_iterator& tree<T, tree_nodeallocator>::leaf_iterator::operator-=(unsigned int num){
		   while(num>0) {
		      --(*this);
		      --num;
		      }
		   return (*this);
		}
    };
};
#endif