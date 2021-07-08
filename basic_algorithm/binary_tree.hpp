#ifndef PROJECT_BASE_ALGORITHM_BINARY_TREE_H
#define PROJECT_BASE_ALGORITHM_BINARY_TREE_H
#include <ProjectBase/basic_algorithm/tree.hpp>
namespace ProjectBase{
    namespace Tree{
        template<typename T>
        class _binary_tree_node: _tree_node<T>{
            public:
                _binary_tree_node();
                _binary_tree_node(const T&);
                _binary_tree_node(T&&);
            public:
                virtual _binary_tree_node<T>* left() const=0;
                virtual _binary_tree_node<T>* right() const=0;
				virtual _binary_tree_node<T>* pre() const=0;
				virtual _binary_tree_node<T>* next() const=0;
				virtual _binary_tree_node<T>* first_child() const=0;
				virtual _binary_tree_node<T>* last_child() const=0;
				virtual _binary_tree_node<T>* parent() const=0;
        };

        template<typename T>
        _binary_tree_node<T>::_binary_tree_node(): _tree_node<T>(){};

        template<typename T>
        _binary_tree_node<T>::_binary_tree_node(const T& lval): _tree_node<T>(lval){};

        template<typename T>
        _binary_tree_node<T>::_binary_tree_node(T&& rval): _tree_node<T>(rval){};

        template<typename T>
        class binary_tree_node: _binary_tree_node<T>{
            public:
                _binary_tree_node<T>* left() const;
                _binary_tree_node<T>* right() const;
            public:
				virtual _binary_tree_node<T>* pre() const;
				virtual _binary_tree_node<T>* next() const;
				virtual _binary_tree_node<T>* first_child() const;
				virtual _binary_tree_node<T>* last_child() const;
				virtual _binary_tree_node<T>* parent() const;
            public:
                binary_tree_node<T>* _left_node;
                binary_tree_node<T>* _right_node;
        };

        template<typename T>
        _binary_tree_node<T>* binary_tree_node<T>::left() const{
            return _left_node;
        };

        template<typename T>
        _binary_tree_node<T>* binary_tree_node<T>::right() const{
            return _right_node;
        };

        template<typename Node>
        class _binary_tree: _tree<Node>{
        };

        /**
         * \brief 模板类
         * \note 
         * \param[in] Node 继承于_binary_node的类
         * \author none
         * \since 0.0.1
         * */
        template<typename Node>
        class BinaryTree: _binary_tree<Node> {
            public:
                BinaryTree();
            public:
				//class in_order_iterator : public iterator_base { 
				//	public:
				//		in_order_iterator();
				//		in_order_iterator(tree_node *);
				//		in_order_iterator(const iterator_base&);
				//		in_order_iterator(const sibling_iterator&);

				//		bool    operator==(const in_order_iterator&) const;
				//		bool    operator!=(const in_order_iterator&) const;
				//		in_order_iterator&  operator++();
				//	   	in_order_iterator&  operator--();
				//		in_order_iterator   operator++(int);
				//		in_order_iterator   operator--(int);
				//		in_order_iterator&  operator+=(unsigned int);
				//		in_order_iterator&  operator-=(unsigned int);

				//		in_order_iterator&  next_skip_children();
				//};
        };

        template<typename Node>
        BinaryTree<Node>::BinaryTree(): _binary_tree<Node>(){

        }
    };
};
#endif