// src/Page_de_Chat.ts
console.log("Page_de_Chat.ts (With History Item Options) loaded");

// --- DOM Elements ---
const sidebar = document.getElementById('sidebar') as HTMLElement | null;
const sidebarToggleBtn = document.getElementById('sidebarToggleBtn') as HTMLButtonElement | null;
const mobileSidebarOverlay = document.getElementById('mobileSidebarOverlay') as HTMLElement | null;
const mainAreaMobileSidebarToggleBtn = document.getElementById('mainAreaMobileSidebarToggleBtn') as HTMLButtonElement | null;
const mobileSidebarCloseBtn = document.getElementById('mobileSidebarCloseBtn') as HTMLButtonElement | null;
const mainChatArea = document.getElementById('mainChatArea') as HTMLElement | null;

const historyToggleBtn = document.getElementById('historyToggleBtn') as HTMLButtonElement | null;
const historyArrowIcon = document.getElementById('historyArrowIcon') as SVGElement | null;
const historySection = document.getElementById('historySection') as HTMLElement |null;
const conversationHistoryList = document.getElementById('conversationHistoryList') as HTMLElement | null;
const searchConversationsInput = document.getElementById('searchConversations') as HTMLInputElement | null;

const messageDisplayArea = document.getElementById('messageDisplayArea') as HTMLElement | null;
const welcomeMessageContainer = document.getElementById('welcomeMessageContainer') as HTMLElement | null;
const typingIndicatorContainer = document.getElementById('typingIndicatorContainer') as HTMLElement | null;
let typingIndicatorElement: HTMLElement | null = null;

const messageInputFooter = document.getElementById('messageInputFooter') as HTMLTextAreaElement | null;
const sendMsgBtnFooter = document.getElementById('sendMsgBtnFooter') as HTMLButtonElement | null;
const uploadFileBtnFooter = document.getElementById('uploadFileBtnFooter') as HTMLButtonElement | null;
const fileInputFooter = document.getElementById('fileInputFooter') as HTMLInputElement | null;

const currentChatTitleHeader = document.getElementById('currentChatTitleHeader') as HTMLElement | null;

const userProfileContainer = document.getElementById('userProfileContainer') as HTMLElement | null;
const userAccountDropdown = document.getElementById('userAccountDropdown') as HTMLElement | null;
const userNameDisplay = document.getElementById('userNameDisplay') as HTMLElement | null;
const userAvatar = document.getElementById('userAvatar') as HTMLImageElement | null;
const logoutChatBtn = document.getElementById('logoutChatBtn') as HTMLAnchorElement | null;
const accountSettingsLink = document.getElementById('accountSettingsLink') as HTMLAnchorElement | null;
const newChatBtn = document.getElementById('newChatBtn') as HTMLButtonElement | null;

// --- State Variables ---
let currentChatId: string | null = null;
let isSidebarProgrammaticallyOpen = window.innerWidth >= 768;
let isHistorySectionOpen = true;
let chatsData = [ // Let's make chatsData modifiable
    { id: "gem_hist_1", name: "Introduction à Gemini", lastMsg: "..." },
    { id: "gem_hist_2", name: "Capacités multimodales", lastMsg: "..." },
    { id: "gem_hist_3", name: "Exemples d'utilisation", lastMsg: "..." },
];


// --- Helper Functions ---
function scrollToBottom() {
    if (messageDisplayArea) {
        setTimeout(() => { messageDisplayArea.scrollTop = messageDisplayArea.scrollHeight; }, 0);
    }
}

function hasActualMessages(): boolean {
    if (!messageDisplayArea) return false;
    const messagesInnerContainer = messageDisplayArea.querySelector('.max-w-3xl');
    if (!messagesInnerContainer) return false;
    for (const child of Array.from(messagesInnerContainer.children)) {
        if (child !== welcomeMessageContainer && child !== typingIndicatorContainer) {
            if (child.classList.contains('message-bubble-user') || child.classList.contains('message-bubble-bot') ||
                child.querySelector('.message-bubble-user') || child.querySelector('.message-bubble-bot')) {
                if (!child.classList.contains('typing-indicator-wrapper') && !child.closest('.typing-indicator-wrapper')) {
                    return true;
                }
            }
        }
    }
    return false;
}

function updateWelcomeMessageVisibility() {
    const hasMessages = hasActualMessages();
    if (welcomeMessageContainer) {
        welcomeMessageContainer.classList.toggle('hidden', hasMessages);
    }
}

function createMessageBubble(message: string, sender: 'user' | 'bot'): HTMLElement {
    const outerDiv = document.createElement('div');
    outerDiv.classList.add('mb-4', 'flex', sender === 'user' ? 'justify-end' : 'justify-start');
    const bubbleDiv = document.createElement('div');
    bubbleDiv.classList.add(
        'p-3', 'rounded-xl', 'max-w-xs', 'sm:max-w-sm', 'md:max-w-md', 'lg:max-w-lg',
        'text-sm', 'shadow-md', 'break-words',
        sender === 'user' ? 'message-bubble-user' : 'message-bubble-bot'
    );
    bubbleDiv.textContent = message;
    outerDiv.appendChild(bubbleDiv);
    return outerDiv;
}

function getTypingIndicator(): HTMLElement {
    if (!typingIndicatorElement) {
        typingIndicatorElement = document.createElement('div');
        typingIndicatorElement.classList.add('mb-4', 'flex', 'justify-start', 'typing-indicator-wrapper');
        typingIndicatorElement.innerHTML = `
            <div class="p-3 rounded-xl message-bubble-bot text-sm shadow-md flex items-center space-x-1">
                <span class="typing-dot animate-bounce" style="animation-delay: 0s;">●</span>
                <span class="typing-dot animate-bounce" style="animation-delay: 0.1s;">●</span>
                <span class="typing-dot animate-bounce" style="animation-delay: 0.2s;">●</span>
            </div>`;
        typingIndicatorElement.style.display = 'none';
        if (typingIndicatorContainer) {
            typingIndicatorContainer.innerHTML = '';
            typingIndicatorContainer.appendChild(typingIndicatorElement);
        } else {
            console.error("Static typing indicator container (#typingIndicatorContainer) not found!");
        }
    }
    return typingIndicatorElement;
}

function showTypingIndicator(show: boolean) {
    const indicatorBubble = getTypingIndicator();
    if (indicatorBubble) {
        indicatorBubble.style.display = show ? 'flex' : 'none';
        if (show) scrollToBottom();
    }
}

function displayMessage(messageElement: HTMLElement) {
    if (!messageDisplayArea) return;
    showTypingIndicator(false);
    const messagesInnerContainer = messageDisplayArea.querySelector('.max-w-3xl');
    if (!messagesInnerContainer) return;
    if (typingIndicatorContainer) {
        messagesInnerContainer.insertBefore(messageElement, typingIndicatorContainer);
    } else {
        messagesInnerContainer.appendChild(messageElement);
    }
    updateWelcomeMessageVisibility();
    scrollToBottom();
}

function clearChatArea() {
    const messagesInnerContainer = messageDisplayArea?.querySelector('.max-w-3xl');
    if (messagesInnerContainer) {
        Array.from(messagesInnerContainer.children).forEach(child => {
            if (child !== welcomeMessageContainer && child !== typingIndicatorContainer) {
                messagesInnerContainer.removeChild(child);
            }
        });
    }
    if (typingIndicatorElement) {
       showTypingIndicator(false);
    }
}

async function sendMessageFooter() {
    if (!messageInputFooter || !sendMsgBtnFooter) return;
    const messageText = messageInputFooter.value.trim();
    if (!messageText) return;
    const userMessageBubble = createMessageBubble(messageText, 'user');
    displayMessage(userMessageBubble);
    messageInputFooter.value = '';
    if(sendMsgBtnFooter) sendMsgBtnFooter.disabled = true;
    messageInputFooter.style.height = 'auto';
    messageInputFooter.focus();
    showTypingIndicator(true);
    console.log(`Sending to backend (Chat ID: ${currentChatId || 'new'}): "${messageText}"`);
    setTimeout(() => {
        const botResponse = `Réponse Gemini simulée pour: "${messageText}".`;
        const botMessageBubble = createMessageBubble(botResponse, 'bot');
        displayMessage(botMessageBubble);
        console.log(`Received from backend (Simulated)`);
    }, 1500 + Math.random() * 1000);
}

function loadChat(chatId: string, chatName: string) {
    console.log(`Loading chat: ${chatId} (${chatName})`);
    currentChatId = chatId;
    if (currentChatTitleHeader) currentChatTitleHeader.textContent = chatName;
    clearChatArea();
    getTypingIndicator();
    showTypingIndicator(true);
    setTimeout(() => {
        const simulatedMessages = [
            createMessageBubble(`Gemini: Bienvenue dans la conversation "${chatName}".`, 'bot'),
            createMessageBubble(`Humain: Ma première question dans "${chatName}"...`, 'user'),
            createMessageBubble(`Gemini: Une réponse pertinente.`, 'bot')
        ];
        simulatedMessages.forEach(displayMessage);
        console.log("Simulated messages loaded for chat:", chatName);
    }, 800);
    if (window.innerWidth < 768) setSidebarState(false);
    if (messageInputFooter) messageInputFooter.focus();
}

function startNewChat() {
    console.log("Starting new Gemini-style chat");
    currentChatId = null;
    if (currentChatTitleHeader) currentChatTitleHeader.textContent = "Nouveau Chat";
    clearChatArea();
    getTypingIndicator();
    showTypingIndicator(false);
    if (messageInputFooter) {
        messageInputFooter.value = '';
        messageInputFooter.style.height = 'auto';
        messageInputFooter.focus();
    }
    if (sendMsgBtnFooter) sendMsgBtnFooter.disabled = true;
    updateWelcomeMessageVisibility();
    if (window.innerWidth < 768) setSidebarState(false);
    document.querySelectorAll('.history-item.selected').forEach(el => el.classList.remove('selected'));
}

if (sendMsgBtnFooter) sendMsgBtnFooter.addEventListener('click', sendMessageFooter);
if (messageInputFooter && sendMsgBtnFooter) {
    messageInputFooter.addEventListener('input', () => {
        messageInputFooter.style.height = 'auto';
        const scrollHeight = messageInputFooter.scrollHeight;
        const maxHeight = 120;
        messageInputFooter.style.height = `${Math.min(scrollHeight, maxHeight)}px`;
        sendMsgBtnFooter.disabled = messageInputFooter.value.trim().length === 0;
    });
    messageInputFooter.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey && !sendMsgBtnFooter.disabled) {
            event.preventDefault();
            sendMessageFooter();
        }
    });
}
if (uploadFileBtnFooter && fileInputFooter) {
    uploadFileBtnFooter.addEventListener('click', () => fileInputFooter.click());
    fileInputFooter.addEventListener('change', () => {
        const files = fileInputFooter.files;
        if (files && files.length > 0) {
            alert(`Fichier sélectionné (Gemini sim): ${files[0].name}`);
            fileInputFooter.value = '';
        }
    });
}
if (newChatBtn) newChatBtn.addEventListener('click', startNewChat);

function setSidebarState(expanded: boolean) {
    if (!sidebar || !sidebarToggleBtn || !mainChatArea) return;
    isSidebarProgrammaticallyOpen = expanded;
    const isMobile = window.innerWidth < 768;
    sidebar.classList.toggle('is-expanded-desktop', !isMobile && expanded);
    sidebar.classList.toggle('is-collapsed-desktop', !isMobile && !expanded);
    sidebar.classList.toggle('is-mobile-visible', isMobile && expanded);
    sidebar.classList.toggle('is-mobile-hidden', isMobile && !expanded);
    mainChatArea.classList.toggle('desktop-sidebar-expanded', !isMobile && expanded);
    mainChatArea.classList.toggle('desktop-sidebar-collapsed', !isMobile && !expanded);
    if (mobileSidebarOverlay) mobileSidebarOverlay.classList.toggle('hidden', !expanded || !isMobile);
    sidebarToggleBtn.setAttribute('aria-label', expanded ? 'Réduire le menu' : 'Étendre le menu');
    sidebarToggleBtn.setAttribute('aria-expanded', expanded.toString());
}
if (sidebarToggleBtn) sidebarToggleBtn.addEventListener('click', () => setSidebarState(!isSidebarProgrammaticallyOpen));
if (mainAreaMobileSidebarToggleBtn) mainAreaMobileSidebarToggleBtn.addEventListener('click', () => { if (window.innerWidth < 768) setSidebarState(true); });
if (mobileSidebarCloseBtn) mobileSidebarCloseBtn.addEventListener('click', () => { if (window.innerWidth < 768) setSidebarState(false); });
if (mobileSidebarOverlay) mobileSidebarOverlay.addEventListener('click', () => { if (window.innerWidth < 768) setSidebarState(false); });

function setHistorySectionState(open: boolean) {
    if (!historySection || !historyArrowIcon || !historyToggleBtn) return;
    isHistorySectionOpen = open;
    if (open) {
        historySection.classList.remove('collapsed');
        historyArrowIcon.classList.add('rotate-180');
        historyToggleBtn.setAttribute('aria-expanded', 'true');
        historySection.style.maxHeight = `${historySection.scrollHeight}px`;
        setTimeout(() => {
            if (isHistorySectionOpen && !historySection.classList.contains('collapsed')) {
                historySection.style.maxHeight = 'none';
            }
        }, 300);
    } else {
        historySection.style.maxHeight = `${historySection.scrollHeight}px`;
        requestAnimationFrame(() => {
            historySection.classList.add('collapsed');
            historyArrowIcon.classList.remove('rotate-180');
            historyToggleBtn.setAttribute('aria-expanded', 'false');
            historySection.style.maxHeight = '0px';
        });
    }
}
if (historyToggleBtn) {
    historyToggleBtn.addEventListener('click', () => setHistorySectionState(!isHistorySectionOpen));
}
if (userProfileContainer && userAccountDropdown) {
    userProfileContainer.addEventListener('click', (event) => {
        event.stopPropagation();
        userAccountDropdown.classList.toggle('hidden');
    });
    document.addEventListener('click', (event) => {
        if (!userAccountDropdown.classList.contains('hidden') &&
            !userProfileContainer.contains(event.target as Node) &&
            !userAccountDropdown.contains(event.target as Node)) {
            userAccountDropdown.classList.add('hidden');
        }
    });
    if (logoutChatBtn) logoutChatBtn.addEventListener('click', (e) => {
        e.preventDefault();
        alert("Déconnexion Gemini (sim).");
        window.location.href = './index.html';
    });
    if (accountSettingsLink) accountSettingsLink.addEventListener('click', (e) => {
        e.preventDefault();
        alert("Paramètres Gemini (sim).");
        if(userAccountDropdown) userAccountDropdown.classList.add('hidden');
    });
}

// --- START: MODIFIED populateExampleConversations and NEW handleHistoryAction ---
function populateExampleConversations() {
    if (!conversationHistoryList) return;
    conversationHistoryList.innerHTML = ''; // Clear existing items

    // Use the global chatsData array
    if (chatsData.length === 0) {
        conversationHistoryList.innerHTML = '<p class="px-4 py-2 text-xs text-gray-500 text-center">Aucun historique.</p>';
        return;
    }

    chatsData.forEach(chat => {
        const historyItemContainer = document.createElement('div');
        historyItemContainer.className = "history-item-container relative group"; // 'group' for hover effects on children

        // Wrapper for the link and the options button to sit side-by-side
        const linkAndButtonWrapper = document.createElement('div');
        linkAndButtonWrapper.className = 'flex items-center w-full p-0.5 rounded-full'; // Added p-0.5 and rounded-full to wrapper for better hover on items

        const itemLink = document.createElement('a');
        itemLink.href = "#";
        itemLink.className = "history-item flex-grow flex items-center text-sm text-gray-700 px-2.5 py-2 rounded-full hover:bg-gray-100 transition-colors duration-150 cursor-pointer min-w-0";
        itemLink.dataset.chatId = chat.id;
        itemLink.dataset.chatName = chat.name;
        itemLink.title = chat.name;

        const nameSpan = document.createElement('span');
        nameSpan.className = "truncate flex-1 sidebar-text"; // flex-1 to take available space, truncate for long names
        nameSpan.textContent = chat.name;
        itemLink.appendChild(nameSpan);

        const optionsButton = document.createElement('button');
        // Styling for the options button (three dots)
        optionsButton.className = "history-item-options-btn p-1.5 rounded-full text-gray-400 hover:text-gray-600 hover:bg-gray-200 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity duration-150 ml-1 flex-shrink-0";
        optionsButton.setAttribute('aria-label', `Options pour ${chat.name}`);
        optionsButton.innerHTML = `
            <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z"></path>
            </svg>
        `;

        const dropdownMenu = document.createElement('div');
        dropdownMenu.className = "history-item-dropdown absolute right-2 top-full mt-1 w-48 bg-white rounded-md shadow-xl ring-1 ring-black ring-opacity-5 py-1 z-30 hidden"; // Increased z-index
        dropdownMenu.innerHTML = `
            <a href="#" class="history-dropdown-action block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900" data-action="rename" data-chat-id="${chat.id}">Renommer</a>
            <a href="#" class="history-dropdown-action block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900" data-action="archive" data-chat-id="${chat.id}">Archiver</a>
            <a href="#" class="history-dropdown-action block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900" data-action="share" data-chat-id="${chat.id}">Partager</a>
            <div class="border-t border-gray-100 my-1"></div>
            <a href="#" class="history-dropdown-action block px-4 py-2 text-sm text-red-600 hover:bg-red-50 hover:text-red-700" data-action="delete" data-chat-id="${chat.id}">Supprimer</a>
        `;

        optionsButton.addEventListener('click', (e) => {
            e.stopPropagation();
            e.preventDefault();
            document.querySelectorAll('.history-item-dropdown:not(.hidden)').forEach(dd => {
                if (dd !== dropdownMenu) {
                    dd.classList.add('hidden');
                }
            });
            dropdownMenu.classList.toggle('hidden');
             // Position dropdown relative to button if needed, though `absolute right-2 top-full` should work with relative parent.
        });

        linkAndButtonWrapper.appendChild(itemLink);
        linkAndButtonWrapper.appendChild(optionsButton);

        historyItemContainer.appendChild(linkAndButtonWrapper);
        historyItemContainer.appendChild(dropdownMenu);

        itemLink.addEventListener('click', (e) => {
            e.preventDefault();
            document.querySelectorAll('.history-item-dropdown:not(.hidden)').forEach(dd => dd.classList.add('hidden'));
            document.querySelectorAll('.history-item.selected').forEach(el => el.classList.remove('selected'));
            itemLink.classList.add('selected'); // Add selected to the link itself
            if (chat.id && chat.name) loadChat(chat.id, chat.name);
        });

        dropdownMenu.querySelectorAll('.history-dropdown-action').forEach(actionLink => {
            actionLink.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const action = (e.currentTarget as HTMLElement).dataset.action;
                const chatId = (e.currentTarget as HTMLElement).dataset.chatId;
                const currentName = chatsData.find(c => c.id === chatId)?.name || chat.name; // Get current name from data
                handleHistoryAction(action, chatId, currentName, historyItemContainer);
                dropdownMenu.classList.add('hidden');
            });
        });
        conversationHistoryList.appendChild(historyItemContainer);
    });
}

function handleHistoryAction(action: string | undefined, chatId: string | undefined, chatName: string | undefined, itemElement: HTMLElement) {
    if (!action || !chatId || !chatName) return;
    console.log(`Action: ${action}, Chat ID: ${chatId}, Chat Name: ${chatName}`);

    switch (action) {
        case 'rename':
            const newName = prompt(`Renommer la conversation "${chatName}":`, chatName);
            if (newName && newName.trim() !== "" && newName !== chatName) {
                // 1. Update data source
                const chatIndex = chatsData.findIndex(c => c.id === chatId);
                if (chatIndex > -1) {
                    chatsData[chatIndex].name = newName;
                }
                // 2. Update UI (can be more robust by re-rendering, but this is direct)
                const nameSpan = itemElement.querySelector('.history-item .sidebar-text');
                if (nameSpan) nameSpan.textContent = newName;
                const itemLink = itemElement.querySelector<HTMLElement>('.history-item');
                if (itemLink) itemLink.dataset.chatName = newName; // Update dataset for future actions
                alert(`Conversation "${chatName}" renommée en "${newName}".`);
                 // If this chat is currently loaded, update the header
                if (currentChatId === chatId && currentChatTitleHeader) {
                    currentChatTitleHeader.textContent = newName;
                }
            }
            break;
        case 'archive':
            alert(`Action: Archiver la conversation "${chatName}" (ID: ${chatId}) - Simulation.`);
            // TODO: Implement actual archive logic (e.g., move to an archived list, update UI)
            // For now, let's remove it from the current list and data
            chatsData = chatsData.filter(c => c.id !== chatId);
            itemElement.remove();
            if (chatsData.length === 0 && conversationHistoryList) {
                 conversationHistoryList.innerHTML = '<p class="px-4 py-2 text-xs text-gray-500 text-center">Aucun historique.</p>';
            }
            break;
        case 'share':
            alert(`Action: Partager la conversation "${chatName}" (ID: ${chatId}) - Simulation.`);
            // TODO: Implement actual share logic
            break;
        case 'delete':
            if (confirm(`Êtes-vous sûr de vouloir supprimer la conversation "${chatName}" ? Cette action est irréversible.`)) {
                // 1. Update data source
                chatsData = chatsData.filter(c => c.id !== chatId);
                // 2. Remove element from DOM
                itemElement.remove();
                alert(`Conversation "${chatName}" supprimée.`);
                // If the deleted chat was the current one, start a new chat
                if (currentChatId === chatId) {
                    startNewChat();
                }
                 if (chatsData.length === 0 && conversationHistoryList) {
                    conversationHistoryList.innerHTML = '<p class="px-4 py-2 text-xs text-gray-500 text-center">Aucun historique.</p>';
                }
            }
            break;
        default:
            console.warn(`Unknown history action: ${action}`);
    }
}
// --- END: MODIFIED populateExampleConversations and NEW handleHistoryAction ---

if (searchConversationsInput && conversationHistoryList) {
    searchConversationsInput.addEventListener('input', () => {
        const searchTerm = searchConversationsInput.value.toLowerCase();
        const items = conversationHistoryList.querySelectorAll<HTMLDivElement>('.history-item-container'); // Search within containers
        let visibleItemsCount = 0;
        items.forEach(itemContainer => {
            const itemLink = itemContainer.querySelector<HTMLAnchorElement>('a.history-item');
            if(itemLink){
                const name = itemLink.dataset.chatName?.toLowerCase() || '';
                const isVisible = name.includes(searchTerm);
                itemContainer.style.display = isVisible ? 'block' : 'none'; // Show/hide the whole container
                if (isVisible) visibleItemsCount++;
            }
        });
        let noResultsEl = conversationHistoryList.querySelector<HTMLElement>('.no-results-message');
        if (visibleItemsCount === 0 && searchTerm.length > 0 && chatsData.length > 0) { // Only show if there was data to search
            if (!noResultsEl) {
                noResultsEl = document.createElement('p');
                noResultsEl.className = 'no-results-message px-4 py-2 text-xs text-gray-500 text-center';
                noResultsEl.textContent = 'Aucun résultat trouvé.';
                conversationHistoryList.appendChild(noResultsEl);
            }
        } else if (noResultsEl) {
            noResultsEl.remove();
        }
    });
}

function initializeChatInterface() {
    const userData = { name: "Utilisateur Gemini", avatarUrl: null };
    if (userNameDisplay) userNameDisplay.textContent = userData.name;
    if (userAvatar) {
        if (userData.avatarUrl) {
            userAvatar.src = userData.avatarUrl;
        } else {
            const nameParts = userData.name.split(' ');
            const initials = (nameParts[0]?.[0] || '') + (nameParts.length > 1 ? nameParts[1]?.[0] || '' : (nameParts[0]?.length > 1 ? nameParts[0][1] : ''));
            userAvatar.src = `https://via.placeholder.com/40/339DFF/ffffff?text=${initials.toUpperCase() || 'U'}`;
            userAvatar.alt = userData.name;
        }
    }
    setSidebarState(isSidebarProgrammaticallyOpen);
    getTypingIndicator();
    populateExampleConversations(); // This will now include the options button logic
    if (historyToggleBtn) {
        setHistorySectionState(isHistorySectionOpen);
    }
    if (messageInputFooter && sendMsgBtnFooter) {
        sendMsgBtnFooter.disabled = messageInputFooter.value.trim().length === 0;
        messageInputFooter.style.height = 'auto';
        messageInputFooter.style.height = `${messageInputFooter.scrollHeight}px`;
    }
    startNewChat();
    console.log("Gemini-style Chat interface initialized (With History Options).");

    // Global click listener to close history dropdowns
    document.addEventListener('click', (e) => {
        const openDropdowns = document.querySelectorAll('.history-item-dropdown:not(.hidden)');
        openDropdowns.forEach(dropdown => {
            // Check if the click was outside the dropdown AND its associated options button
            const optionsBtn = dropdown.previousElementSibling?.querySelector('.history-item-options-btn');
            if (!dropdown.contains(e.target as Node) && (optionsBtn && !optionsBtn.contains(e.target as Node))) {
                dropdown.classList.add('hidden');
            }
        });
    });
}

document.addEventListener('DOMContentLoaded', initializeChatInterface);

window.addEventListener('resize', () => {
    const isDesktop = window.innerWidth >= 768;
    const shouldBeOpen = isDesktop ? isSidebarProgrammaticallyOpen : false;
    const currentMobileState = sidebar?.classList.contains('is-mobile-visible');
    const currentDesktopState = sidebar?.classList.contains('is-expanded-desktop');
    if (isDesktop) {
        if (shouldBeOpen !== currentDesktopState) {
            setSidebarState(shouldBeOpen);
        }
    } else {
        if (currentMobileState && !shouldBeOpen) {
            setSidebarState(false);
        }
    }
    if (isHistorySectionOpen && historySection && !historySection.classList.contains('collapsed')) {
        historySection.style.maxHeight = 'none';
        requestAnimationFrame(() => {
            if(isHistorySectionOpen && historySection && !historySection.classList.contains('collapsed')) {
                historySection.style.maxHeight = `${historySection.scrollHeight}px`;
                setTimeout(() => {
                    if (isHistorySectionOpen && historySection && !historySection.classList.contains('collapsed')) {
                        historySection.style.maxHeight = 'none';
                    }
                }, 300);
            }
        });
    }
});
export {};